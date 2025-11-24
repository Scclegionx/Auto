package com.auto_fe.auto_fe.models

data class ChatRoom(
    val id: Long,
    val user1Id: Long,
    val user1Name: String?,
    val user1Avatar: String?,
    val user2Id: Long,
    val user2Name: String?,
    val user2Avatar: String?,
    val lastMessage: String?,
    val lastMessageTime: String?,
    val unreadCount: Long,
    val createdAt: String?,
    val updatedAt: String?
)

data class ChatMessage(
    val id: Long,
    val chatId: Long,
    val senderId: Long,
    val senderName: String?,
    val senderAvatar: String?,
    val content: String,
    val isRead: Boolean,
    val readAt: String?,
    val createdAt: String?
)

data class SendMessageRequest(
    val chatId: Long? = null,
    val receiverId: Long? = null,
    val content: String
)
