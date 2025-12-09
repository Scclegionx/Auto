package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
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
import com.auto_fe.auto_fe.ui.service.RelationshipService
import com.auto_fe.auto_fe.ui.service.UserService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

/**
 * Màn hình danh sách Elder cho Supervisor
 * Hiển thị tất cả Elder mà Supervisor đang giám sát (đã kết nối ACCEPTED)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ElderListScreen(
    accessToken: String,
    userAvatar: String? = null,
    onElderClick: (Long, String) -> Unit = { _, _ -> },  // Updated signature to pass elderUserName
    onSearchUserClick: () -> Unit = {},
    onChatClick: () -> Unit = {},
    onProfileClick: () -> Unit = {},
    onNotificationHistoryClick: () -> Unit = {},
    onManageConnectionsClick: () -> Unit = {},
    onLogout: () -> Unit = {}
) {
    val scope = rememberCoroutineScope()
    val relationshipService = remember { RelationshipService() }
    
    var elderRelationships by remember { mutableStateOf<List<RelationshipService.RelationshipRequest>>(emptyList()) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showMenu by remember { mutableStateOf(false) }
    
    // Load danh sách Elder đã kết nối (ACCEPTED)
    LaunchedEffect(Unit) {
        isLoading = true
        scope.launch {
            try {
                relationshipService.getConnectedElders(accessToken).onSuccess { relationships ->
                    elderRelationships = relationships
                    isLoading = false
                }.onFailure { error ->
                    errorMessage = error.message
                    isLoading = false
                }
            } catch (e: Exception) {
                errorMessage = e.message
                isLoading = false
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Người được giám sát",
                            fontSize = AppTextSize.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = DarkOnPrimary
                        )
                        Text(
                            "${elderRelationships.size} người",
                            fontSize = AppTextSize.bodySmall,
                            color = DarkOnPrimary.copy(alpha = 0.8f)
                        )
                    }
                },
                actions = {
                    // Avatar button (thay icon 3 chấm)
                    Box(
                        modifier = Modifier
                            .padding(end = 8.dp)
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(DarkOnPrimary.copy(alpha = 0.2f))
                            .clickable { showMenu = !showMenu },
                        contentAlignment = Alignment.Center
                    ) {
                        if (!userAvatar.isNullOrBlank()) {
                            AsyncImage(
                                model = userAvatar,
                                contentDescription = "Avatar",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(CircleShape)
                            )
                        } else {
                            Icon(
                                imageVector = Icons.Default.Person,
                                contentDescription = "Menu",
                                tint = DarkOnPrimary,
                                modifier = Modifier.size(24.dp)
                            )
                        }
                    }
                    
                    DropdownMenu(
                        expanded = showMenu,
                        onDismissRequest = { showMenu = false },
                        modifier = Modifier.background(DarkSurface)
                    ) {
                        DropdownMenuItem(
                            text = {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Icon(Icons.Default.Notifications, null, tint = DarkOnSurface, modifier = Modifier.size(20.dp))
                                    Text("Lịch sử thông báo", color = DarkOnSurface)
                                }
                            },
                            onClick = {
                                showMenu = false
                                onNotificationHistoryClick()
                            }
                        )
                        
                        DropdownMenuItem(
                            text = {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Icon(Icons.Default.Person, null, tint = DarkOnSurface, modifier = Modifier.size(20.dp))
                                    Text("Thông tin cá nhân", color = DarkOnSurface)
                                }
                            },
                            onClick = {
                                showMenu = false
                                onProfileClick()
                            }
                        )
                        
                        DropdownMenuItem(
                            text = {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Icon(Icons.Default.SupervisedUserCircle, null, tint = DarkOnSurface, modifier = Modifier.size(20.dp))
                                    Text("Quản lý yêu cầu", color = DarkOnSurface)
                                }
                            },
                            onClick = {
                                showMenu = false
                                onManageConnectionsClick()
                            }
                        )
                        
                        Divider(color = DarkOnSurface.copy(alpha = 0.2f))
                        
                        DropdownMenuItem(
                            text = {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Icon(Icons.Default.Logout, null, tint = DarkError, modifier = Modifier.size(20.dp))
                                    Text("Đăng xuất", color = DarkError)
                                }
                            },
                            onClick = {
                                showMenu = false
                                onLogout()
                            }
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkPrimary,
                    titleContentColor = DarkOnPrimary
                )
            )
        },
        containerColor = DarkBackground,
        floatingActionButton = {
            // FAB cho Chat và Thêm Elder
            Column(
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.End
            ) {
                // Chat FAB
                FloatingActionButton(
                    onClick = onChatClick,
                    containerColor = DarkPrimary,
                    contentColor = DarkOnPrimary
                ) {
                    Icon(Icons.Default.Chat, contentDescription = "Tin nhắn")
                }
                
                // Add Elder FAB
                FloatingActionButton(
                    onClick = onSearchUserClick,
                    containerColor = DarkPrimary,
                    contentColor = DarkOnPrimary
                ) {
                    Icon(Icons.Default.PersonAdd, contentDescription = "Thêm Elder")
                }
            }
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            when {
                isLoading -> {
                    // Loading state
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator(color = DarkPrimary)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                "Đang tải danh sách...",
                                color = DarkOnSurface.copy(alpha = 0.6f)
                            )
                        }
                    }
                }
                
                errorMessage != null -> {
                    // Error state
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(24.dp)
                        ) {
                            Icon(
                                Icons.Default.Error,
                                contentDescription = null,
                                tint = DarkError,
                                modifier = Modifier.size(64.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                errorMessage ?: "Đã xảy ra lỗi",
                                color = DarkOnSurface,
                                fontSize = AppTextSize.bodyMedium
                            )
                        }
                    }
                }
                
                elderRelationships.isEmpty() -> {
                    // Empty state
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            modifier = Modifier.padding(24.dp)
                        ) {
                            Icon(
                                Icons.Default.PersonOff,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.3f),
                                modifier = Modifier.size(80.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                "Chưa có người được giám sát",
                                color = DarkOnSurface.copy(alpha = 0.6f),
                                fontSize = AppTextSize.bodyLarge,
                                fontWeight = FontWeight.Medium
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                "Nhấn vào nút + để thêm người cần giám sát",
                                color = DarkOnSurface.copy(alpha = 0.4f),
                                fontSize = AppTextSize.bodySmall
                            )
                            Spacer(modifier = Modifier.height(24.dp))
                            Button(
                                onClick = onSearchUserClick,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = DarkPrimary
                                )
                            ) {
                                Icon(Icons.Default.PersonAdd, null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Thêm người giám sát")
                            }
                        }
                    }
                }
                
                else -> {
                    // Elder list
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(elderRelationships, key = { it.elderUserId }) { relationship ->
                            ElderCard(
                                relationship = relationship,
                                onClick = { onElderClick(relationship.elderUserId, relationship.elderUserName) }  // Pass both ID and name
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ElderCard(
    relationship: RelationshipService.RelationshipRequest,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface
        ),
        elevation = CardDefaults.cardElevation(4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.Top
            ) {
                // Avatar
                Box(
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(DarkPrimary.copy(alpha = 0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    if (relationship.elderUserAvatar != null) {
                        AsyncImage(
                            model = relationship.elderUserAvatar,
                            contentDescription = "Avatar",
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(CircleShape)
                        )
                    } else {
                        Icon(
                            Icons.Default.Person,
                            contentDescription = null,
                            tint = DarkPrimary,
                            modifier = Modifier.size(36.dp)
                        )
                    }
                }
                
                // Info
                Column(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    // Name
                    Text(
                        relationship.elderUserName.ifEmpty { "Chưa có tên" },
                        fontSize = AppTextSize.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    
                    // Email
                    if (relationship.elderUserEmail != null) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                Icons.Default.Email,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                relationship.elderUserEmail,
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnSurface.copy(alpha = 0.7f),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                    
                    // Phone
                    if (relationship.elderUserPhone != null) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Icon(
                                Icons.Default.Phone,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.6f),
                                modifier = Modifier.size(16.dp)
                            )
                            Text(
                                relationship.elderUserPhone,
                                fontSize = AppTextSize.bodySmall,
                                color = DarkOnSurface.copy(alpha = 0.7f)
                            )
                        }
                    }
                }
                
                // Arrow
                Icon(
                    Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint = DarkPrimary,
                    modifier = Modifier.size(28.dp)
                )
            }
            
            // Health info section
            if (relationship.elderBloodType != null || relationship.elderHeight != null || relationship.elderWeight != null || relationship.elderGender != null) {
                Spacer(modifier = Modifier.height(12.dp))
                Divider(color = DarkOnSurface.copy(alpha = 0.1f), thickness = 1.dp)
                Spacer(modifier = Modifier.height(12.dp))
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Blood Type
                    relationship.elderBloodType?.let {
                        HealthInfoChip(
                            icon = Icons.Default.Favorite,
                            label = it,
                            color = Color(0xFFE91E63)
                        )
                    }
                    
                    // Height
                    relationship.elderHeight?.let {
                        HealthInfoChip(
                            icon = Icons.Default.Height,
                            label = "${it.toInt()} cm",
                            color = Color(0xFF2196F3)
                        )
                    }
                    
                    // Weight
                    relationship.elderWeight?.let {
                        HealthInfoChip(
                            icon = Icons.Default.MonitorWeight,
                            label = "${it.toInt()} kg",
                            color = Color(0xFF4CAF50)
                        )
                    }
                    
                    // Gender
                    relationship.elderGender?.let { gender ->
                        HealthInfoChip(
                            icon = if (gender == "MALE") Icons.Default.Male else Icons.Default.Female,
                            label = when(gender) {
                                "MALE" -> "Nam"
                                "FEMALE" -> "Nữ"
                                else -> "Khác"
                            },
                            color = if (gender == "MALE") Color(0xFF03A9F4) else Color(0xFFE91E63)
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun HealthInfoChip(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    color: Color = DarkPrimary
) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = color.copy(alpha = 0.12f)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(5.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                icon,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(16.dp)
            )
            Text(
                label,
                fontSize = AppTextSize.labelSmall,
                color = color,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
