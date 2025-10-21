package com.auto_fe.auto_fe.ui.service

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Service để xử lý authentication với Auto_BE API
 */
class AuthService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    // TODO: Thay đổi URL này thành URL server của bạn
    private val baseUrl = "http://192.168.33.103:8080/api/auth" // For Android Emulator
    // private val baseUrl = "http://YOUR_IP:8080/api/auth" // For Real Device

    /**
     * Response data classes
     */
    data class LoginResponse(
        val status: String,
        val message: String,
        val data: LoginData?
    )

    data class LoginData(
        val accessToken: String
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

    /**
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
                        LoginData(
                            accessToken = dataJson.getString("accessToken")
                        )
                    } else null

                    Result.success(LoginResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Login failed: ${response.code}")
                        } catch (e: Exception) {
                            "Login failed: ${response.code}"
                        }
                    } else {
                        "Login failed: ${response.code}"
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
                            errorJson.optString("message", "Register failed: ${response.code}")
                        } catch (e: Exception) {
                            "Register failed: ${response.code}"
                        }
                    } else {
                        "Register failed: ${response.code}"
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
                val json = JSONObject().apply {
                    put("fcmToken", fcmToken)
                    put("deviceId", deviceId)
                    put("deviceType", deviceType)
                    put("deviceName", deviceName)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("${baseUrl.replace("/auth", "")}/device-token/register")
                    .post(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

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
                            errorJson.optString("message", "Register device token failed: ${response.code}")
                        } catch (e: Exception) {
                            "Register device token failed: ${response.code}"
                        }
                    } else {
                        "Register device token failed: ${response.code}"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("AuthService", "Register device token error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
