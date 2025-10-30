package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

/**
 * Service để xử lý authentication với Auto_BE API
 */
class AuthService {
    private val client by lazy { ApiClient.getClient() }

    // Lấy baseUrl từ ApiConfig
    private val baseUrl = com.auto_fe.auto_fe.network.ApiConfig.BASE_URL + "/auth"

    /**
     * Response data classes
     */
    data class LoginResponse(
        val status: String,
        val message: String,
        val data: LoginData?
    )

    data class LoginData(
        val accessToken: String,
        val user: UserInfo?
    )
    
    data class UserInfo(
        val id: Long?,
        val email: String?,
        val name: String?
    )

    data class RegisterResponse(
        val status: String,
        val message: String,
        val data: Any?
    )

    data class DeviceTokenResponse(
        val status: String,
        val message: String,
        val data: DeviceTokenData?
    )
    
    data class DeviceTokenData(
        val id: String,
        val fcmToken: String,
        val deviceId: String,
        val deviceType: String,
        val deviceName: String
    )
    
    data class VerificationResponse(
        val status: String,
        val message: String,
        val data: Any?
    )    /**
     * Đăng nhập với email và password
     * @param email Email của user
     * @param password Password của user
     * @return Result<LoginResponse> - Success nếu đăng nhập thành công, Failure nếu có lỗi
     */
    suspend fun login(email: String, password: String): Result<LoginResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("email", email)
                    put("password", password)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/login")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("AuthService", "Login Response Code: ${response.code}")
                Log.d("AuthService", "Login Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        
                        // Parse user info if available
                        val userInfo = if (dataJson.has("user") && !dataJson.isNull("user")) {
                            val userJson = dataJson.getJSONObject("user")
                            UserInfo(
                                id = if (userJson.has("id")) userJson.getLong("id") else null,
                                email = userJson.optString("email", null),
                                name = userJson.optString("name", null)
                            )
                        } else null
                        
                        LoginData(
                            accessToken = dataJson.getString("accessToken"),
                            user = userInfo
                        )
                    } else null

                    Result.success(LoginResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Đăng nhập thất bại")
                        } catch (e: Exception) {
                            "Đăng nhập thất bại"
                        }
                    } else {
                        "Đăng nhập thất bại"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Login error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Đăng ký tài khoản mới
     * @param email Email của user
     * @param password Password của user
     * @return Result<RegisterResponse> - Success nếu đăng ký thành công, Failure nếu có lỗi
     */
    suspend fun register(email: String, password: String): Result<RegisterResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("email", email)
                    put("password", password)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/register")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("AuthService", "Register Response Code: ${response.code}")
                Log.d("AuthService", "Register Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    Result.success(RegisterResponse(status, message, null))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Đăng ký thất bại")
                        } catch (e: Exception) {
                            "Đăng ký thất bại"
                        }
                    } else {
                        "Đăng ký thất bại"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Register error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Đăng ký device token sau khi đăng nhập
     * @param fcmToken Firebase Cloud Messaging token
     * @param deviceId Device ID
     * @param deviceType Device type (Android/iOS)
     * @param deviceName Device name
     * @param accessToken JWT access token từ login
     * @return Result<DeviceTokenResponse> - Success nếu đăng ký thành công, Failure nếu có lỗi
     */
    suspend fun registerDeviceToken(
        fcmToken: String,
        deviceId: String,
        deviceType: String,
        deviceName: String,
        accessToken: String
    ): Result<DeviceTokenResponse> {
        return withContext(Dispatchers.IO) {
            try {
                Log.d("AuthService", "=== STARTING registerDeviceToken ===")
                Log.d("AuthService", "fcmToken: ${fcmToken.take(20)}...")
                Log.d("AuthService", "deviceId: $deviceId")
                Log.d("AuthService", "deviceType: $deviceType")
                Log.d("AuthService", "deviceName: $deviceName")
                Log.d("AuthService", "accessToken: ${accessToken.take(20)}...")
                
                val json = JSONObject().apply {
                    put("fcmToken", fcmToken)
                    put("deviceId", deviceId)
                    put("deviceType", deviceType)
                    put("deviceName", deviceName)
                }
                
                Log.d("AuthService", "Request JSON: ${json.toString()}")

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val url = "${baseUrl.replace("/auth", "")}/device-token/register"
                Log.d("AuthService", "Request URL: $url")
                
                val request = Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()
                
                Log.d("AuthService", "Sending HTTP POST request...")

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("AuthService", "Register Device Token Response Code: ${response.code}")
                Log.d("AuthService", "Register Device Token Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        DeviceTokenData(
                            id = dataJson.optString("id", ""),
                            fcmToken = dataJson.optString("fcmToken", ""),
                            deviceId = dataJson.optString("deviceId", ""),
                            deviceType = dataJson.optString("deviceType", ""),
                            deviceName = dataJson.optString("deviceName", "")
                        )
                    } else null

                    Result.success(DeviceTokenResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Đăng ký thiết bị thất bại")
                        } catch (e: Exception) {
                            "Đăng ký thiết bị thất bại"
                        }
                    } else {
                        "Đăng ký thiết bị thất bại"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Register device token error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Gửi mã OTP xác thực email
     */
    suspend fun sendVerificationOtp(email: String): Result<VerificationResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("email", email)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/send-verification-otp")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("AuthService", "Send OTP Response Code: ${response.code}")
                Log.d("AuthService", "Send OTP Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(VerificationResponse(status, message, null))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Gửi mã OTP thất bại")
                        } catch (e: Exception) {
                            "Gửi mã OTP thất bại"
                        }
                    } else {
                        "Gửi mã OTP thất bại"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Send OTP error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Xác thực mã OTP
     */
    suspend fun verifyOtp(email: String, otp: String): Result<VerificationResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("email", email)
                    put("otp", otp)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/verify-otp")
                    .post(requestBody)
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("AuthService", "Verify OTP Response Code: ${response.code}")
                Log.d("AuthService", "Verify OTP Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(VerificationResponse(status, message, null))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Xác thực OTP thất bại")
                        } catch (e: Exception) {
                            "Xác thực OTP thất bại"
                        }
                    } else {
                        "Xác thực OTP thất bại"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Verify OTP error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
