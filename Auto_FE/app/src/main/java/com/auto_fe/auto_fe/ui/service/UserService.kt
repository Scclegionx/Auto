package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Request
import org.json.JSONObject

/**
 * Service để xử lý user profile với Auto_BE API
 */
class UserService {
    private val client by lazy { ApiClient.getClient() }

    // TODO: Thay đổi URL này thành URL server của bạn
    private val baseUrl = "http://192.168.33.102:8080/api/users"

    /**
     * Response data classes
     */
    data class ProfileResponse(
        val status: String,
        val message: String,
        val data: ProfileData?
    )

    data class ProfileData(
        val id: Long?,
        val fullName: String?,
        val email: String?,
        val dateOfBirth: String?, // LocalDate từ backend
        val gender: String?, // MALE, FEMALE, OTHER
        val phoneNumber: String?,
        val address: String?,
        val bloodType: String?, // A, B, AB, O, A_POSITIVE, etc.
        val height: Double?,
        val weight: Double?,
        val avatar: String?,
        val isActive: Boolean?
    )

    /**
     * Lấy thông tin profile của user hiện tại
     */
    suspend fun getUserProfile(accessToken: String): Result<ProfileResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/profile")
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("UserService", "Get Profile Response Code: ${response.code}")
                Log.d("UserService", "Get Profile Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        ProfileData(
                            id = dataJson.optLong("id"),
                            fullName = dataJson.optString("fullName", null),
                            email = dataJson.optString("email", null),
                            dateOfBirth = dataJson.optString("dateOfBirth", null),
                            gender = dataJson.optString("gender", null),
                            phoneNumber = dataJson.optString("phoneNumber", null),
                            address = dataJson.optString("address", null),
                            bloodType = dataJson.optString("bloodType", null),
                            height = if (dataJson.has("height") && !dataJson.isNull("height")) 
                                dataJson.getDouble("height") else null,
                            weight = if (dataJson.has("weight") && !dataJson.isNull("weight")) 
                                dataJson.getDouble("weight") else null,
                            avatar = dataJson.optString("avatar", null),
                            isActive = dataJson.optBoolean("isActive", false)
                        )
                    } else null

                    Result.success(ProfileResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể tải thông tin profile")
                        } catch (e: Exception) {
                            "Không thể tải thông tin profile"
                        }
                    } else {
                        "Không thể tải thông tin profile"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("UserService", "Get profile error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
