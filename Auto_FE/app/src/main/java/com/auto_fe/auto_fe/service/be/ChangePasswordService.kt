package com.auto_fe.auto_fe.service.be

import android.util.Log
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import org.json.JSONObject

/**
 * Service để xử lý đổi mật khẩu với Auto_BE API
 */
class ChangePasswordService {
    private val client by lazy { ApiClient.getClient() }
    private val baseUrl = com.auto_fe.auto_fe.config.be.BeConfig.BASE_URL + "/users"

    data class ChangePasswordResponse(
        val status: String,
        val message: String,
        val data: String?
    )

    /**
     * Đổi mật khẩu (yêu cầu mật khẩu cũ)
     */
    suspend fun changePassword(
        accessToken: String,
        currentPassword: String,
        newPassword: String
    ): Result<ChangePasswordResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("currentPassword", currentPassword)
                    put("newPassword", newPassword)
                }

                Log.d("ChangePasswordService", "Change Password Request: $jsonBody")

                val requestBody = jsonBody.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

                val request = Request.Builder()
                    .url("$baseUrl/change-password")
                    .put(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ChangePasswordService", "Response Code: ${response.code}")
                Log.d("ChangePasswordService", "Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    val data = jsonResponse.optString("data", null)
                    
                    Result.success(ChangePasswordResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể đổi mật khẩu")
                        } catch (e: Exception) {
                            "Không thể đổi mật khẩu"
                        }
                    } else {
                        "Không thể đổi mật khẩu"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("ChangePasswordService", "Change Password Error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
