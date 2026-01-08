package com.auto_fe.auto_fe.service.be

import android.util.Log
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Request
import org.json.JSONObject
import org.json.JSONArray

class UserGuideService {
    private val client by lazy { ApiClient.getClient() }
    private val baseUrl = com.auto_fe.auto_fe.config.be.BeConfig.BASE_URL + "/user-guides"

    data class GuideResponse(
        val status: String,
        val message: String,
        val data: List<GuideItem>?
    )

    data class GuideItem(
        val id: Long?,
        val title: String?,
        val description: String?,
        val videoUrl: String?,
        val thumbnailUrl: String?,
        val userType: String?,
        val displayOrder: Int?,
        val createdAt: String?,
        val updatedAt: String?
    )

    suspend fun getGuides(): Result<GuideResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val url = "$baseUrl/elder"

                Log.d("UserGuideService", "GetGuides URL: $url")

                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("UserGuideService", "GetGuides Response Code: ${response.code}")
                Log.d("UserGuideService", "GetGuides Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        val guidesList = mutableListOf<GuideItem>()

                        for (i in 0 until dataArray.length()) {
                            val itemJson = dataArray.getJSONObject(i)
                            guidesList.add(
                                GuideItem(
                                    id = if (itemJson.has("id")) itemJson.getLong("id") else null,
                                    title = itemJson.optString("title", null),
                                    description = itemJson.optString("description", null),
                                    videoUrl = itemJson.optString("videoUrl", null),
                                    thumbnailUrl = itemJson.optString("thumbnailUrl", null),
                                    userType = itemJson.optString("userType", null),
                                    displayOrder = if (itemJson.has("displayOrder")) itemJson.getInt("displayOrder") else null,
                                    createdAt = itemJson.optString("createdAt", null),
                                    updatedAt = itemJson.optString("updatedAt", null)
                                )
                            )
                        }
                        guidesList.sortedBy { it.displayOrder ?: Int.MAX_VALUE }
                    } else null

                    Result.success(GuideResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể tải danh sách hướng dẫn")
                        } catch (e: Exception) {
                            "Không thể tải danh sách hướng dẫn"
                        }
                    } else {
                        "Không thể tải danh sách hướng dẫn"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("UserGuideService", "GetGuides error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}

