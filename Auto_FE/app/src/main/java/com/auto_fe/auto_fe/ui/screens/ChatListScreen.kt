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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.auto_fe.auto_fe.models.ChatRoom
import com.auto_fe.auto_fe.ui.service.ChatService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatListScreen(
    accessToken: String,
    currentUserId: Long,
    onChatClick: (Long) -> Unit,
    onSearchUserClick: () -> Unit = {},
    onBackClick: () -> Unit
) {
    var chats by remember { mutableStateOf<List<ChatRoom>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()
    val chatService = remember { ChatService() }

    LaunchedEffect(Unit) {
        scope.launch {
            try {
                val result = chatService.getAllChats(accessToken)
                result.onSuccess { chatList ->
                    chats = chatList
                }.onFailure { error ->
                    errorMessage = "Không thể tải danh sách chat: ${error.message}"
                }
            } catch (e: Exception) {
                errorMessage = "Lỗi: ${e.message}"
            } finally {
                isLoading = false
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(AIBackgroundDeep, AIBackgroundSoft)
                )
            )
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            // Top App Bar
            TopAppBar(
                title = {
                    Text(
                        "Tin nhắn",
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Quay lại",
                            tint = DarkOnSurface
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onSearchUserClick) {
                        Icon(
                            Icons.Default.Search,
                            contentDescription = "Tìm người dùng",
                            tint = DarkOnSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface.copy(alpha = 0.9f)
                )
            )

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
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                Icons.Default.Warning,
                                contentDescription = null,
                                tint = DarkError,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                errorMessage ?: "Có lỗi xảy ra",
                                color = DarkOnSurface,
                                fontSize = AppTextSize.bodyMedium
                            )
                        }
                    }
                }
                chats.isEmpty() -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Icon(
                                Icons.Default.MailOutline,
                                contentDescription = null,
                                tint = DarkOnSurface.copy(alpha = 0.5f),
                                modifier = Modifier.size(64.dp)
                            )
                            Text(
                                "Chưa có cuộc trò chuyện nào",
                                color = DarkOnSurface.copy(alpha = 0.7f),
                                fontSize = AppTextSize.bodyLarge,
                                fontWeight = FontWeight.Medium
                            )
                            Text(
                                "Bắt đầu chat mới để liên lạc",
                                color = DarkOnSurface.copy(alpha = 0.5f),
                                fontSize = AppTextSize.bodyMedium
                            )
                        }
                    }
                }
                else -> {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        items(chats) { chat ->
                            ChatRoomCard(
                                chat = chat,
                                currentUserId = currentUserId,
                                onClick = { onChatClick(chat.id) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatRoomCard(
    chat: ChatRoom,
    currentUserId: Long,
    onClick: () -> Unit
) {
    // Xác định người kia (other user)
    val isCurrentUserUser1 = chat.user1Id == currentUserId
    val otherUserName = if (isCurrentUserUser1) chat.user2Name else chat.user1Name
    val otherUserAvatar = if (isCurrentUserUser1) chat.user2Avatar else chat.user1Avatar

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.8f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Avatar
            Box(
                modifier = Modifier
                    .size(56.dp)
                    .clip(CircleShape)
                    .background(DarkPrimary.copy(alpha = 0.2f)),
                contentAlignment = Alignment.Center
            ) {
                if (!otherUserAvatar.isNullOrBlank()) {
                    AsyncImage(
                        model = otherUserAvatar,
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
                        modifier = Modifier.size(32.dp)
                    )
                }
            }

            // Content
            Column(
                modifier = Modifier
                    .weight(1f)
                    .align(Alignment.CenterVertically),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = otherUserName ?: "Người dùng",
                        fontSize = AppTextSize.bodyLarge,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )

                    if (chat.unreadCount > 0) {
                        Badge(
                            containerColor = DarkPrimary
                        ) {
                            Text(
                                text = if (chat.unreadCount > 99) "99+" else chat.unreadCount.toString(),
                                color = DarkOnPrimary,
                                fontSize = 12.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                Text(
                    text = chat.lastMessage ?: "Chưa có tin nhắn",
                    fontSize = AppTextSize.bodyMedium,
                    color = DarkOnSurface.copy(alpha = 0.6f),
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )

                if (chat.lastMessageTime != null) {
                    Text(
                        text = formatTime(chat.lastMessageTime),
                        fontSize = AppTextSize.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.5f)
                    )
                }
            }
        }
    }
}

private fun formatTime(isoString: String): String {
    return try {
        val inputFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.getDefault())
        val outputFormat = SimpleDateFormat("HH:mm, dd/MM", Locale.getDefault())
        val date = inputFormat.parse(isoString)
        outputFormat.format(date ?: Date())
    } catch (e: Exception) {
        "Vừa xong"
    }
}
