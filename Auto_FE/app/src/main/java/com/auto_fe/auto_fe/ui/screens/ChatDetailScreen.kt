package com.auto_fe.auto_fe.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.models.ChatMessage
import com.auto_fe.auto_fe.ui.service.ChatService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatDetailScreen(
    accessToken: String,
    currentUserId: Long,
    chatId: Long? = null,        // Chat ID (luôn có sau khi create)
    receiverId: Long? = null,    // Deprecated - không dùng nữa
    chatName: String? = null,
    onBackClick: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val chatService = remember { ChatService() }
    val listState = rememberLazyListState()

    var messages by remember { mutableStateOf<List<ChatMessage>>(emptyList()) }
    var messageText by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Load messages if chatId exists
    LaunchedEffect(chatId) {
        chatId?.let { id ->
            scope.launch {
                isLoading = true
                errorMessage = null
                val result = chatService.getMessages(id, 0, 50, accessToken)
                result.fold(
                    onSuccess = { pageResponse ->
                        messages = pageResponse.content.reversed() // Reverse to show newest at bottom
                        isLoading = false
                        // Scroll to bottom
                        if (messages.isNotEmpty()) {
                            listState.animateScrollToItem(messages.size - 1)
                        }
                    },
                    onFailure = { error ->
                        errorMessage = "Không thể tải tin nhắn: ${error.message}"
                        isLoading = false
                    }
                )
            }
        }
    }

    // Send message function
    val sendMessage: () -> Unit = {
        if (messageText.isNotBlank() && chatId != null) {
            scope.launch {
                val result = chatService.sendMessage(
                    chatId = chatId,
                    receiverId = null, // Không cần receiverId nữa
                    content = messageText.trim(),
                    accessToken = accessToken
                )
                result.fold(
                    onSuccess = { sentMessage ->
                        // Add message to list
                        messages = messages + sentMessage
                        messageText = ""
                        // Scroll to bottom
                        scope.launch {
                            listState.animateScrollToItem(messages.size - 1)
                        }
                    },
                    onFailure = { error ->
                        errorMessage = "Không thể gửi tin nhắn: ${error.message}"
                        Log.e("ChatDetailScreen", "Send message failed", error)
                    }
                )
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        chatName ?: "Chat",
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = DarkOnSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface.copy(alpha = 0.9f)
                )
            )
        },
        containerColor = AIBackgroundDeep
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Messages List
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                when {
                    isLoading -> {
                        CircularProgressIndicator(
                            modifier = Modifier.align(Alignment.Center),
                            color = DarkPrimary
                        )
                    }
                    errorMessage != null -> {
                        Text(
                            text = errorMessage ?: "",
                            color = AIError,
                            modifier = Modifier
                                .align(Alignment.Center)
                                .padding(16.dp)
                        )
                    }
                    messages.isEmpty() -> {
                        Text(
                            text = "Chưa có tin nhắn.\nGửi tin nhắn đầu tiên!",
                            color = DarkOnSurface.copy(alpha = 0.6f),
                            modifier = Modifier
                                .align(Alignment.Center)
                                .padding(16.dp),
                            textAlign = androidx.compose.ui.text.style.TextAlign.Center
                        )
                    }
                    else -> {
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.fillMaxSize(),
                            contentPadding = PaddingValues(16.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(messages) { message ->
                                MessageBubble(
                                    message = message,
                                    isCurrentUser = message.senderId == currentUserId
                                )
                            }
                        }
                    }
                }
            }

            // Input Area
            Card(
                modifier = Modifier
                    .fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface
                ),
                shape = RoundedCornerShape(0.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedTextField(
                        value = messageText,
                        onValueChange = { messageText = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Nhập tin nhắn...") },
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedContainerColor = AIBackgroundSoft,
                            unfocusedContainerColor = AIBackgroundSoft,
                            focusedTextColor = DarkOnSurface,
                            unfocusedTextColor = DarkOnSurface,
                            focusedBorderColor = DarkPrimary,
                            unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                            cursorColor = DarkPrimary
                        ),
                        shape = RoundedCornerShape(24.dp),
                        maxLines = 5
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = sendMessage,
                        enabled = messageText.isNotBlank()
                    ) {
                        Icon(
                            Icons.Default.Send,
                            contentDescription = "Send",
                            tint = if (messageText.isNotBlank()) DarkPrimary else DarkOnSurface.copy(alpha = 0.3f)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MessageBubble(
    message: ChatMessage,
    isCurrentUser: Boolean
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isCurrentUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            modifier = Modifier
                .widthIn(max = 280.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (isCurrentUser) DarkPrimary else DarkSurface
            ),
            shape = RoundedCornerShape(
                topStart = 16.dp,
                topEnd = 16.dp,
                bottomStart = if (isCurrentUser) 16.dp else 4.dp,
                bottomEnd = if (isCurrentUser) 4.dp else 16.dp
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(12.dp)
            ) {
                Text(
                    text = message.content,
                    color = if (isCurrentUser) DarkOnPrimary else DarkOnSurface,
                    fontSize = AppTextSize.bodyMedium
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = formatTime(message.createdAt),
                    color = if (isCurrentUser) 
                        DarkOnPrimary.copy(alpha = 0.7f) 
                    else 
                        DarkOnSurface.copy(alpha = 0.6f),
                    fontSize = AppTextSize.bodySmall
                )
            }
        }
    }
}

private fun formatTime(timestamp: String?): String {
    if (timestamp == null) return ""
    return try {
        val instant = java.time.Instant.parse(timestamp)
        val localTime = java.time.LocalDateTime.ofInstant(
            instant,
            java.time.ZoneId.systemDefault()
        )
        val formatter = java.time.format.DateTimeFormatter.ofPattern("HH:mm")
        localTime.format(formatter)
    } catch (e: Exception) {
        Log.e("ChatDetailScreen", "Error formatting time: $timestamp", e)
        ""
    }
}
