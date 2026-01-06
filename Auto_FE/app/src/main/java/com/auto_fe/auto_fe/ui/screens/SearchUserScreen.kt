package com.auto_fe.auto_fe.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.service.be.UserService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchUserScreen(
    accessToken: String,
    onUserClick: (Long, String) -> Unit, // userId, userName
    onBackClick: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val userService = remember { UserService() }

    var searchQuery by remember { mutableStateOf("") }
    var users by remember { mutableStateOf<List<UserService.ProfileData>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Load all users on start
    LaunchedEffect(Unit) {
        scope.launch {
            isLoading = true
            errorMessage = null
            val result = userService.searchUsers(accessToken, null)
            result.fold(
                onSuccess = { userList ->
                    users = userList
                    isLoading = false
                },
                onFailure = { error ->
                    errorMessage = error.message
                    isLoading = false
                }
            )
        }
    }

    // Search when query changes
    LaunchedEffect(searchQuery) {
        if (searchQuery.isNotBlank()) {
            scope.launch {
                isLoading = true
                errorMessage = null
                val result = userService.searchUsers(accessToken, searchQuery)
                result.fold(
                    onSuccess = { userList ->
                        users = userList
                        isLoading = false
                    },
                    onFailure = { error ->
                        errorMessage = error.message
                        isLoading = false
                    }
                )
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Tìm người dùng") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Back")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface
                )
            )
        },
        containerColor = AIBackgroundDeep
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            // Search Bar
            OutlinedTextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("Tìm theo tên hoặc email...") },
                leadingIcon = {
                    Icon(Icons.Default.Search, contentDescription = "Search")
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = DarkSurface,
                    unfocusedContainerColor = DarkSurface,
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface,
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    cursorColor = DarkPrimary
                ),
                shape = RoundedCornerShape(12.dp),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Content
            when {
                isLoading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = DarkPrimary)
                    }
                }
                errorMessage != null -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = errorMessage ?: "Có lỗi xảy ra",
                            color = AIError
                        )
                    }
                }
                users.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "Không tìm thấy người dùng",
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                }
                else -> {
                    LazyColumn(
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(users) { user ->
                            UserCard(
                                user = user,
                                onClick = {
                                    user.id?.let { userId ->
                                        onUserClick(userId, user.fullName ?: user.email ?: "User")
                                    }
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun UserCard(
    user: UserService.ProfileData,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Avatar
            if (user.avatar != null) {
                AsyncImage(
                    model = user.avatar,
                    contentDescription = "Avatar",
                    modifier = Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary.copy(alpha = 0.2f))
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(48.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary.copy(alpha = 0.2f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = (user.fullName ?: user.email ?: "U").first().uppercase(),
                        color = DarkPrimary,
                        fontWeight = FontWeight.Bold,
                        fontSize = AppTextSize.titleMedium
                    )
                }
            }

            // User Info
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = user.fullName ?: "Không có tên",
                    fontWeight = FontWeight.Bold,
                    fontSize = AppTextSize.bodyLarge,
                    color = DarkOnSurface,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = user.email ?: "",
                    fontSize = AppTextSize.bodySmall,
                    color = DarkOnSurface.copy(alpha = 0.6f),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}
