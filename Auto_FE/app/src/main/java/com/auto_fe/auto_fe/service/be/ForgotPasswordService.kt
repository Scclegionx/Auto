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
 * Service để xử lý quên mật khẩu với Auto_BE API
 */
class ForgotPasswordService {
    private val client by lazy { ApiClient.getClient() }
    private val baseUrl = com.auto_fe.auto_fe.config.be.BeConfig.BASE_URL + "/auth"

    data class ForgotPasswordResponse(
        val status: String,
        val message: String
    )

    /**
     * Bước 1: Gửi OTP cho quên mật khẩu
     */
    suspend fun sendForgotPasswordOtp(email: String): Result<ForgotPasswordResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("email", email)
                }

                Log.d("ForgotPasswordService", "Send Forgot Password OTP Request: $jsonBody")

                val requestBody = jsonBody.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

                val request = Request.Builder()
                    .url("$baseUrl/send-forgot-password-otp")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ForgotPasswordService", "Response Code: ${response.code}")
                Log.d("ForgotPasswordService", "Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(ForgotPasswordResponse(status, message))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không tìm thấy email")
                        } catch (e: Exception) {
                            "Không tìm thấy email"
                        }
                    } else {
                        "Không tìm thấy email"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("ForgotPasswordService", "Error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Bước 2: Verify OTP cho quên mật khẩu
     */
    suspend fun verifyForgotPasswordOtp(email: String, otp: String): Result<ForgotPasswordResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("email", email)
                    put("otp", otp)
                }

                Log.d("ForgotPasswordService", "Verify Forgot Password OTP Request: $jsonBody")

                val requestBody = jsonBody.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

                val request = Request.Builder()
                    .url("$baseUrl/verify-forgot-password-otp")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ForgotPasswordService", "Response Code: ${response.code}")
                Log.d("ForgotPasswordService", "Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(ForgotPasswordResponse(status, message))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Mã OTP không đúng")
                        } catch (e: Exception) {
                            "Mã OTP không đúng"
                        }
                    } else {
                        "Mã OTP không đúng"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("ForgotPasswordService", "Error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Bước 3: Reset mật khẩu (sau khi verify OTP thành công)
     */
    suspend fun forgotPassword(email: String): Result<ForgotPasswordResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("email", email)
                }

                Log.d("ForgotPasswordService", "Request: $jsonBody")

                val requestBody = jsonBody.toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaTypeOrNull())

                val request = Request.Builder()
                    .url("$baseUrl/forgot-password")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("ForgotPasswordService", "Response Code: ${response.code}")
                Log.d("ForgotPasswordService", "Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(ForgotPasswordResponse(status, message))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không tìm thấy email")
                        } catch (e: Exception) {
                            "Không tìm thấy email"
                        }
                    } else {
                        "Không tìm thấy email"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("ForgotPasswordService", "Error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
