package com.auto_fe.auto_fe.service.be

import android.util.Log
import com.auto_fe.auto_fe.models.ChatMessage
import com.auto_fe.auto_fe.models.ChatRoom
import com.auto_fe.auto_fe.network.ApiClient
import com.auto_fe.auto_fe.config.be.BeConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.ByteArrayOutputStream
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit
import kotlin.math.max
import okio.buffer
import okio.source
import okio.BufferedSink
import okhttp3.MediaType.Companion.toMediaTypeOrNull
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
    private val baseUrl = BeConfig.BASE_URL + "/chat"

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
        accessToken: String,
        messageType: String = "TEXT",
        attachmentUrl: String? = null,
        attachmentName: String? = null,
        attachmentType: String? = null,
        attachmentSize: Long? = null
    ): Result<ChatMessage> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    if (chatId != null) put("chatId", chatId)
                    if (receiverId != null) put("receiverId", receiverId)
                    put("content", content)
                    put("messageType", messageType)
                    if (attachmentUrl != null) put("attachmentUrl", attachmentUrl)
                    if (attachmentName != null) put("attachmentName", attachmentName)
                    if (attachmentType != null) put("attachmentType", attachmentType)
                    if (attachmentSize != null) put("attachmentSize", attachmentSize)
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
     * Upload ảnh lên server
     */
    suspend fun uploadImage(imageUri: android.net.Uri, context: android.content.Context, accessToken: String): Result<Map<String, Any>> {
        val uploadId = System.currentTimeMillis()
        Log.d("ChatService", "[UPLOAD-START] ID=$uploadId, URI=$imageUri")
        
        return withContext(Dispatchers.IO) {
            try {
                val contentResolver = context.contentResolver

                // Compress image before upload (reduce size/time)
                val compressedBytes = compressImage(context, imageUri)
                    ?: return@withContext Result.failure(Exception("Cannot compress image"))

                Log.d("ChatService", "[UPLOAD-COMPRESS] ID=$uploadId, Compressed size: ${compressedBytes.size / 1024}KB")

                val fileName = getFileName(context, imageUri) ?: "image_${System.currentTimeMillis()}.jpg"
                val mimeType = "image/jpeg" // Always JPEG after compression

                val requestBody = okhttp3.MultipartBody.Builder()
                    .setType(okhttp3.MultipartBody.FORM)
                    .addFormDataPart(
                        "file",
                        fileName,
                        compressedBytes.toRequestBody(mimeType.toMediaType())
                    )
                    .build()

                val request = Request.Builder()
                    .url("${BeConfig.BASE_URL}/upload/image")
                    .post(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                Log.d("ChatService", "[UPLOAD-REQUEST] ID=$uploadId, Sending request to server...")

                // Execute with retry on timeout using extended client
                val response = try {
                    val resp = client.newCall(request).execute()
                    Log.d("ChatService", "[UPLOAD-RESPONSE] ID=$uploadId, First attempt response: ${resp.code}")
                    resp
                } catch (e: SocketTimeoutException) {
                    Log.w("ChatService", "[UPLOAD-TIMEOUT] ID=$uploadId, Timed out, retrying with longer timeouts...")
                    val longClient = client.newBuilder()
                        .readTimeout(180, TimeUnit.SECONDS)
                        .writeTimeout(180, TimeUnit.SECONDS)
                        .build()
                    val retryResp = longClient.newCall(request).execute()
                    Log.d("ChatService", "[UPLOAD-RETRY-RESPONSE] ID=$uploadId, Retry response: ${retryResp.code}")
                    retryResp
                }
                val responseBody = response.body?.string()
                
                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val data = jsonResponse.getJSONObject("data")
                    
                    val result = mutableMapOf<String, Any>()
                    result["url"] = data.getString("url")
                    // If backend didn't return bytes, use compressed size
                    result["bytes"] = if (data.has("bytes") && !data.isNull("bytes")) data.getLong("bytes") else compressedBytes.size.toLong()
                    result["format"] = data.optString("format", "")
                    
                    Log.d("ChatService", "[UPLOAD-SUCCESS] ID=$uploadId, URL=${result["url"]}")
                    Result.success(result as Map<String, Any>)
                } else {
                    Log.e("ChatService", "[UPLOAD-FAIL] ID=$uploadId, Response: ${response.code}, Body: $responseBody")
                    Result.failure(Exception("Upload failed: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "[UPLOAD-ERROR] ID=$uploadId, Error: ${e.message}", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Upload file lên server
     */
    suspend fun uploadFile(fileUri: android.net.Uri, context: android.content.Context, accessToken: String): Result<Map<String, Any>> {
        return withContext(Dispatchers.IO) {
            try {
                val contentResolver = context.contentResolver
                val fileName = getFileName(context, fileUri) ?: "file_${System.currentTimeMillis()}"
                val mimeType = contentResolver.getType(fileUri) ?: "application/octet-stream"

                // Build a streaming RequestBody to avoid loading entire file into memory
                val streamRequestBody = object : okhttp3.RequestBody() {
                    override fun contentType() = mimeType.toMediaTypeOrNull()

                    override fun writeTo(sink: BufferedSink) {
                        // Open the stream here so it is available during actual write
                        contentResolver.openInputStream(fileUri)?.use { input ->
                            val source = input.source().buffer()
                            sink.writeAll(source)
                        } ?: throw java.io.IOException("Cannot open file stream")
                    }
                }

                val requestBody = okhttp3.MultipartBody.Builder()
                    .setType(okhttp3.MultipartBody.FORM)
                    .addFormDataPart(
                        "file",
                        fileName,
                        streamRequestBody
                    )
                    .build()

                val request = Request.Builder()
                    .url("${BeConfig.BASE_URL}/upload/file")
                    .post(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                // Execute with retry on timeout
                val response = try {
                    client.newCall(request).execute()
                } catch (e: SocketTimeoutException) {
                    Log.w("ChatService", "Upload file timed out, retrying with longer timeouts...")
                    val longClient = client.newBuilder()
                        .readTimeout(180, TimeUnit.SECONDS)
                        .writeTimeout(180, TimeUnit.SECONDS)
                        .build()
                    longClient.newCall(request).execute()
                }
                val responseBody = response.body?.string()
                
                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val data = jsonResponse.getJSONObject("data")
                    
                    val result = mutableMapOf<String, Any>()
                    result["url"] = data.getString("url")
                    result["bytes"] = data.getLong("bytes")
                    result["format"] = data.optString("format", "")
                    result["originalFilename"] = data.optString("originalFilename", fileName)
                    
                    Result.success(result as Map<String, Any>)
                } else {
                    Result.failure(Exception("Upload failed: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("ChatService", "Error uploading file", e)
                Result.failure(e)
            }
        }
    }
    
    /**
     * Helper: Lấy tên file từ URI
     */
    fun getFileName(context: android.content.Context, uri: android.net.Uri): String? {
        var fileName: String? = null
        if (uri.scheme == "content") {
            val cursor = context.contentResolver.query(uri, null, null, null, null)
            cursor?.use {
                if (it.moveToFirst()) {
                    val nameIndex = it.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (nameIndex >= 0) {
                        fileName = it.getString(nameIndex)
                    }
                }
            }
        }
        if (fileName == null) {
            fileName = uri.path
            val cut = fileName?.lastIndexOf('/')
            if (cut != null && cut != -1) {
                fileName = fileName?.substring(cut + 1)
            }
        }
        return fileName
    }

    /**
     * Compress image from given Uri to JPEG byte array.
     * Returns null on failure.
     */
    private fun compressImage(context: android.content.Context, uri: android.net.Uri, maxDimension: Int = 1280, quality: Int = 80): ByteArray? {
        try {
            val resolver = context.contentResolver

            // First decode bounds
            var input = resolver.openInputStream(uri) ?: return null
            val boundsOptions = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeStream(input, null, boundsOptions)
            input.close()

            val origW = boundsOptions.outWidth
            val origH = boundsOptions.outHeight
            if (origW <= 0 || origH <= 0) return null

            // Compute inSampleSize
            var inSampleSize = 1
            val maxOrig = max(origW, origH)
            if (maxOrig > maxDimension) {
                inSampleSize = Math.floor(maxOrig.toDouble() / maxDimension.toDouble()).toInt()
                if (inSampleSize < 1) inSampleSize = 1
            }

            val opts = BitmapFactory.Options().apply { this.inSampleSize = inSampleSize }
            input = resolver.openInputStream(uri) ?: return null
            var bitmap = BitmapFactory.decodeStream(input, null, opts)
            input.close()

            if (bitmap == null) return null

            // Scale down further if needed
            val maxSide = max(bitmap.width, bitmap.height)
            if (maxSide > maxDimension) {
                val ratio = maxDimension.toFloat() / maxSide.toFloat()
                val newW = (bitmap.width * ratio).toInt()
                val newH = (bitmap.height * ratio).toInt()
                val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
                if (scaled != bitmap) {
                    bitmap.recycle()
                    bitmap = scaled
                }
            }

            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, quality, baos)
            val bytes = baos.toByteArray()
            baos.close()
            bitmap.recycle()
            return bytes
        } catch (e: Exception) {
            Log.e("ChatService", "Failed to compress image", e)
            return null
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
            messageType = json.optString("messageType", "TEXT"),
            attachmentUrl = json.optString("attachmentUrl", null),
            attachmentName = json.optString("attachmentName", null),
            attachmentType = json.optString("attachmentType", null),
            attachmentSize = if (json.has("attachmentSize") && !json.isNull("attachmentSize")) {
                json.getLong("attachmentSize")
            } else null,
            isRead = json.getBoolean("isRead"),
            readAt = json.optString("readAt", null),
            createdAt = json.optString("createdAt", null)
        )
    }
}
