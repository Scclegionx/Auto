package com.auto_fe.auto_fe.ui.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Notification Data Model theo entity mới
 */
data class NotificationData(
    val id: Long,
    val notificationType: String, // MEDICATION_REMINDER, ELDER_MISSED_MEDICATION, etc.
    val title: String,
    val body: String,
    val status: String, // SENT, FAILED
    val isRead: Boolean,
    val actionUrl: String?, // Deep link
    val relatedElderId: Long?, // For supervisor notifications
    val relatedMedicationLogId: Long?, // Link to medication log
    val userId: Long,
    val userEmail: String,
    val createdAt: String,
    val updatedAt: String?
)

/**
 * Notification Service theo nghiệp vụ mới
 * API endpoints:
 * - GET /api/notifications (all notifications với pagination)
 * - GET /api/notifications/unread
 * - GET /api/notifications/read
 * - GET /api/notifications/unread-count
 * - PUT /api/notifications/{id}/read
 * - PUT /api/notifications/read-all
 */
class NotificationService {
    private val client = OkHttpClient()
    private val baseUrl = com.auto_fe.auto_fe.network.ApiConfig.BASE_URL + "/notifications"

    /**
     * Lấy tất cả notifications với phân trang
     * GET /api/notifications?page=0&size=20
     */
    suspend fun getAllNotifications(
        accessToken: String,
        page: Int = 0,
        size: Int = 20
    ): Result<List<NotificationData>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl?page=$page&size=$size")
                .addHeader("Authorization", "Bearer $accessToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val dataArray = jsonResponse.getJSONArray("data")
                val notifications = parseNotificationArray(dataArray)
                
                Result.success(notifications)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Lấy notifications chưa đọc
     * GET /api/notifications/unread?page=0&size=20
     */
    suspend fun getUnreadNotifications(
        accessToken: String,
        page: Int = 0,
        size: Int = 20
    ): Result<List<NotificationData>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/unread?page=$page&size=$size")
                .addHeader("Authorization", "Bearer $accessToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val dataArray = jsonResponse.getJSONArray("data")
                val notifications = parseNotificationArray(dataArray)
                
                Result.success(notifications)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Lấy notifications đã đọc
     * GET /api/notifications/read?page=0&size=20
     */
    suspend fun getReadNotifications(
        accessToken: String,
        page: Int = 0,
        size: Int = 20
    ): Result<List<NotificationData>> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/read?page=$page&size=$size")
                .addHeader("Authorization", "Bearer $accessToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val dataArray = jsonResponse.getJSONArray("data")
                val notifications = parseNotificationArray(dataArray)
                
                Result.success(notifications)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Đếm số thông báo chưa đọc
     * GET /api/notifications/unread-count
     */
    suspend fun getUnreadCount(
        accessToken: String
    ): Result<Long> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/unread-count")
                .addHeader("Authorization", "Bearer $accessToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val count = jsonResponse.getLong("data")
                
                Result.success(count)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Đánh dấu notification đã đọc
     * PUT /api/notifications/{id}/read
     */
    suspend fun markAsRead(
        accessToken: String,
        notificationId: Long
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/$notificationId/read")
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("Content-Type", "application/json")
                .put("".toRequestBody())
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val message = jsonResponse.optString("message", "Đánh dấu đã đọc thành công")
                
                Result.success(message)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Đánh dấu tất cả notifications đã đọc
     * PUT /api/notifications/read-all
     */
    suspend fun markAllAsRead(
        accessToken: String
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/read-all")
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("Content-Type", "application/json")
                .put("".toRequestBody())
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val message = jsonResponse.optString("message", "Đã đánh dấu tất cả đã đọc")
                
                Result.success(message)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Parse JSONArray to List<NotificationData>
     */
    private fun parseNotificationArray(jsonArray: JSONArray): List<NotificationData> {
        val notifications = mutableListOf<NotificationData>()
        
        for (i in 0 until jsonArray.length()) {
            val jsonObject = jsonArray.getJSONObject(i)
            
            notifications.add(
                NotificationData(
                    id = jsonObject.getLong("id"),
                    notificationType = jsonObject.getString("notificationType"),
                    title = jsonObject.getString("title"),
                    body = jsonObject.optString("body", ""),
                    status = jsonObject.getString("status"),
                    isRead = jsonObject.getBoolean("isRead"),
                    actionUrl = jsonObject.optString("actionUrl", null),
                    relatedElderId = if (jsonObject.has("relatedElderId") && !jsonObject.isNull("relatedElderId")) 
                        jsonObject.getLong("relatedElderId") else null,
                    relatedMedicationLogId = if (jsonObject.has("relatedMedicationLogId") && !jsonObject.isNull("relatedMedicationLogId"))
                        jsonObject.getLong("relatedMedicationLogId") else null,
                    userId = jsonObject.getLong("userId"),
                    userEmail = jsonObject.getString("userEmail"),
                    createdAt = jsonObject.getString("createdAt"),
                    updatedAt = jsonObject.optString("updatedAt", null)
                )
            )
        }
        
        return notifications
    }

    /**
     * Format timestamp cho hiển thị
     */
    fun formatTimestamp(timestamp: String): String {
        return try {
            val dateTime = LocalDateTime.parse(timestamp, DateTimeFormatter.ISO_DATE_TIME)
            val now = LocalDateTime.now()
            
            when {
                dateTime.toLocalDate() == now.toLocalDate() -> {
                    // Hôm nay: "10:30"
                    dateTime.format(DateTimeFormatter.ofPattern("HH:mm"))
                }
                dateTime.toLocalDate() == now.toLocalDate().minusDays(1) -> {
                    // Hôm qua: "Hôm qua 10:30"
                    "Hôm qua ${dateTime.format(DateTimeFormatter.ofPattern("HH:mm"))}"
                }
                dateTime.year == now.year -> {
                    // Cùng năm: "15/03 10:30"
                    dateTime.format(DateTimeFormatter.ofPattern("dd/MM HH:mm"))
                }
                else -> {
                    // Khác năm: "15/03/2024"
                    dateTime.format(DateTimeFormatter.ofPattern("dd/MM/yyyy"))
                }
            }
        } catch (e: Exception) {
            timestamp
        }
    }

    /**
     * Get notification type display name (Vietnamese)
     */
    fun getNotificationTypeDisplayName(type: String): String {
        return when (type) {
            "MEDICATION_REMINDER" -> "Nhắc uống thuốc"
            "ELDER_MISSED_MEDICATION" -> "Bỏ lỡ uống thuốc"
            "ELDER_LATE_MEDICATION" -> "Uống thuốc trễ"
            "ELDER_ADHERENCE_LOW" -> "Tuân thủ thấp"
            "ELDER_HEALTH_ALERT" -> "Cảnh báo sức khỏe"
            "SYSTEM_ANNOUNCEMENT" -> "Thông báo hệ thống"
            "RELATIONSHIP_REQUEST" -> "Yêu cầu kết nối"
            "RELATIONSHIP_ACCEPTED" -> "Chấp nhận kết nối"
            else -> type
        }
    }
}
