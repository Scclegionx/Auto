package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.models.ChatMessage
import com.auto_fe.auto_fe.models.ChatRoom
import com.auto_fe.auto_fe.network.ApiClient
import com.auto_fe.auto_fe.network.ApiConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

/**
 * Service để xử lý Chat với Auto_BE API
 */
class ChatService {
    private val client by lazy { ApiClient.getClient() }
    private val baseUrl = ApiConfig.BASE_URL + "/chat"

    /**
     * Data class cho response
     */
    data class BaseResponse<T>(
        val status: String,
        val message: String,
        val data: T?
    )

    data class MessagePageResponse(
        val content: List<ChatMessage>,
        val totalPages: Int,
        val totalElements: Long
    )

    /**
     * Lấy danh sách tất cả chat của user
     */
    suspend fun getAllChats(accessToken: String): Result<List<ChatRoom>> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(baseUrl)
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ChatService", "Get All Chats Response Code: ${response.code}")
                Log.d("ChatService", "Get All Chats Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonArray = JSONArray(responseBody)
                    val chats = mutableListOf<ChatRoom>()

                    for (i in 0 until jsonArray.length()) {
                        val chatJson = jsonArray.getJSONObject(i)
                        chats.add(parseChatRoom(chatJson))
                    }

                    Result.success(chats)
                } else {
                    Result.failure(Exception("Failed to load chats: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error getting all chats", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Lấy chi tiết 1 chat
     */
    suspend fun getChatById(chatId: Long, accessToken: String): Result<ChatRoom> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/$chatId")
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val chatJson = JSONObject(responseBody)
                    Result.success(parseChatRoom(chatJson))
                } else {
                    Result.failure(Exception("Failed to load chat: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error getting chat by id", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Lấy danh sách message trong chat
     */
    suspend fun getMessages(
        chatId: Long,
        page: Int = 0,
        size: Int = 50,
        accessToken: String
    ): Result<MessagePageResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val url = "$baseUrl/$chatId/messages?page=$page&size=$size"
                val request = Request.Builder()
                    .url(url)
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val json = JSONObject(responseBody)
                    val contentArray = json.getJSONArray("content")
                    val messages = mutableListOf<ChatMessage>()

                    for (i in 0 until contentArray.length()) {
                        val msgJson = contentArray.getJSONObject(i)
                        messages.add(parseChatMessage(msgJson))
                    }

                    val pageResponse = MessagePageResponse(
                        content = messages,
                        totalPages = json.getInt("totalPages"),
                        totalElements = json.getLong("totalElements")
                    )
                    Result.success(pageResponse)
                } else {
                    Result.failure(Exception("Failed to load messages: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error getting messages", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Gửi tin nhắn
     */
    suspend fun sendMessage(
        chatId: Long? = null,
        receiverId: Long? = null,
        content: String,
        accessToken: String
    ): Result<ChatMessage> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    if (chatId != null) put("chatId", chatId)
                    if (receiverId != null) put("receiverId", receiverId)
                    put("content", content)
                }

                val requestBody = jsonBody.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/send")
                    .post(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .addHeader("Content-Type", "application/json")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ChatService", "Send Message Response Code: ${response.code}")
                Log.d("ChatService", "Send Message Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val messageJson = JSONObject(responseBody)
                    Result.success(parseChatMessage(messageJson))
                } else {
                    Result.failure(Exception("Failed to send message: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error sending message", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Đánh dấu messages là đã đọc
     */
    suspend fun markAsRead(chatId: Long, accessToken: String): Result<Unit> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/$chatId/read")
                    .put("".toRequestBody())
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()

                if (response.isSuccessful) {
                    Result.success(Unit)
                } else {
                    Result.failure(Exception("Failed to mark as read: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error marking as read", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Tìm hoặc tạo chat với user
     * Trả về chatId để navigate đến ChatDetailScreen
     */
    suspend fun getOrCreateChatWithUser(
        receiverId: Long,
        accessToken: String
    ): Result<Long> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/create?receiverId=$receiverId")
                    .post("".toRequestBody())
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ChatService", "Create Chat Response Code: ${response.code}")
                Log.d("ChatService", "Create Chat Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val baseResponse = JSONObject(responseBody)
                    
                    // Kiểm tra status
                    val status = baseResponse.optString("status", "")
                    if (status != "success") {
                        val errorMsg = baseResponse.optString("message", "Unknown error")
                        return@withContext Result.failure(Exception("Backend error: $errorMsg"))
                    }
                    
                    // Lấy data
                    if (!baseResponse.has("data")) {
                        return@withContext Result.failure(Exception("Response không có data"))
                    }
                    
                    val dataJson = baseResponse.getJSONObject("data")
                    if (!dataJson.has("id")) {
                        return@withContext Result.failure(Exception("Response data không có id. Data: $dataJson"))
                    }
                    
                    val chatId = dataJson.getLong("id")
                    Result.success(chatId)
                } else {
                    Result.failure(Exception("Failed to create chat: ${response.code} - $responseBody"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error creating chat", e)
                Result.failure(e)
            }
        }
    }

    /**
     * Parse JSON to ChatRoom
     */
    private fun parseChatRoom(json: JSONObject): ChatRoom {
        return ChatRoom(
            id = json.getLong("id"),
            user1Id = json.optLong("user1Id", 0L),
            user1Name = json.optString("user1Name", null),
            user1Avatar = json.optString("user1Avatar", null),
            user2Id = json.optLong("user2Id", 0L),
            user2Name = json.optString("user2Name", null),
            user2Avatar = json.optString("user2Avatar", null),
            lastMessage = json.optString("lastMessage", null),
            lastMessageTime = json.optString("lastMessageTime", null),
            unreadCount = json.optLong("unreadCount", 0L),
            createdAt = json.optString("createdAt", null),
            updatedAt = json.optString("updatedAt", null)
        )
    }

    /**
     * Parse JSON to ChatMessage
     */
    private fun parseChatMessage(json: JSONObject): ChatMessage {
        return ChatMessage(
            id = json.getLong("id"),
            chatId = json.getLong("chatId"),
            senderId = json.getLong("senderId"),
            senderName = json.optString("senderName", null),
            senderAvatar = json.optString("senderAvatar", null),
            content = json.getString("content"),
            isRead = json.getBoolean("isRead"),
            readAt = json.optString("readAt", null),
            createdAt = json.optString("createdAt", null)
        )
    }
}
