package com.auto_fe.auto_fe.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

data class NotificationHistoryResponse(
    val id: Long,
    val reminderTime: String,
    val lastSentTime: String?,
    val status: String,
    val isRead: Boolean,
    val title: String?,
    val body: String?,
    val medicationCount: Int,
    val medicationIds: String?,
    val medicationNames: String?,
    val userId: Long,
    val userEmail: String
)

class NotificationHistoryService {
    private val client = OkHttpClient()
    private val baseUrl = com.auto_fe.auto_fe.network.ApiConfig.BASE_URL + "/notifications"

    /**
     * Lấy lịch sử thông báo với filter
     * @param accessToken JWT token
     * @param startDate Ngày bắt đầu (optional, format: yyyy-MM-dd'T'HH:mm:ss)
     * @param endDate Ngày kết thúc (optional)
     * @param status Trạng thái: SENT, FAILED (optional)
     * @param limit Giới hạn số lượng kết quả (default: 50)
     */
    suspend fun getHistory(
        accessToken: String,
        startDate: String? = null,
        endDate: String? = null,
        status: String? = null,
        limit: Int = 50
    ): Result<List<NotificationHistoryResponse>> = withContext(Dispatchers.IO) {
        try {
            // Build URL with query params
            val urlBuilder = StringBuilder("$baseUrl/history?")
            
            startDate?.let { urlBuilder.append("startDate=$it&") }
            endDate?.let { urlBuilder.append("endDate=$it&") }
            status?.let { urlBuilder.append("status=$it&") }
            
            val url = urlBuilder.toString().trimEnd('&', '?')
            
            val request = Request.Builder()
                .url(url)
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
                
                val notifications = mutableListOf<NotificationHistoryResponse>()
                
                // Giới hạn số lượng kết quả
                val maxItems = minOf(dataArray.length(), limit)
                
                for (i in 0 until maxItems) {
                    val item = dataArray.getJSONObject(i)
                    notifications.add(
                        NotificationHistoryResponse(
                            id = item.getLong("id"),
                            reminderTime = item.getString("reminderTime"),
                            lastSentTime = item.optString("lastSentTime", null),
                            status = item.getString("status"),
                            isRead = item.getBoolean("isRead"),
                            title = item.optString("title", null),
                            body = item.optString("body", null),
                            medicationCount = item.getInt("medicationCount"),
                            medicationIds = item.optString("medicationIds", null),
                            medicationNames = item.optString("medicationNames", null),
                            userId = item.getLong("userId"),
                            userEmail = item.getString("userEmail")
                        )
                    )
                }
                
                Result.success(notifications)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Lấy lịch sử hôm nay
     */
    suspend fun getTodayHistory(accessToken: String): Result<List<NotificationHistoryResponse>> = 
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/history/today")
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
     * Lấy lịch sử 7 ngày gần nhất
     */
    suspend fun getWeekHistory(accessToken: String): Result<List<NotificationHistoryResponse>> = 
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/history/week")
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
     * Đánh dấu thông báo đã đọc
     */
    suspend fun markAsRead(
        accessToken: String,
        notificationId: Long
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/$notificationId/read")
                .addHeader("Authorization", "Bearer $accessToken")
                .put(okhttp3.RequestBody.create(null, ByteArray(0)))
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                Result.success(jsonResponse.getString("message"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Đánh dấu tất cả đã đọc
     */
    suspend fun markAllAsRead(accessToken: String): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/read-all")
                .addHeader("Authorization", "Bearer $accessToken")
                .put(okhttp3.RequestBody.create(null, ByteArray(0)))
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception("HTTP ${response.code}: $responseBody")
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                Result.success(jsonResponse.getString("message"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Lấy số thông báo chưa đọc
     */
    suspend fun getUnreadCount(accessToken: String): Result<Long> = withContext(Dispatchers.IO) {
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
                Result.success(jsonResponse.getLong("data"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private fun parseNotificationArray(dataArray: JSONArray): List<NotificationHistoryResponse> {
        val notifications = mutableListOf<NotificationHistoryResponse>()
        
        for (i in 0 until dataArray.length()) {
            val item = dataArray.getJSONObject(i)
            notifications.add(
                NotificationHistoryResponse(
                    id = item.getLong("id"),
                    reminderTime = item.getString("reminderTime"),
                    lastSentTime = item.optString("lastSentTime", null),
                    status = item.getString("status"),
                    isRead = item.getBoolean("isRead"),
                    title = item.optString("title", null),
                    body = item.optString("body", null),
                    medicationCount = item.getInt("medicationCount"),
                    medicationIds = item.optString("medicationIds", null),
                    medicationNames = item.optString("medicationNames", null),
                    userId = item.getLong("userId"),
                    userEmail = item.getString("userEmail")
                )
            )
        }
        
        return notifications
    }
}
