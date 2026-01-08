package com.auto_fe.auto_fe.service.be

import android.util.Log
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import org.json.JSONArray

/**
 * Service để xử lý settings với Auto_BE API
 */
class UserSettingService {
    private val client by lazy { ApiClient.getClient() }

    private val baseUrl = com.auto_fe.auto_fe.config.be.BeConfig.BASE_URL + "/settings"

    /**
     * Response data classes
     */
    data class SettingsByTypeResponse(
        val status: String,
        val message: String,
        val data: List<SettingItem>?
    )

    data class SettingItem(
        val id: Long?,
        val name: String?,
        val description: String?,
        val settingKey: String?,
        val settingType: String?,
        val defaultValue: String?,
        val possibleValues: String?,
        val isActive: Boolean?
    )

    data class UserSettingsResponse(
        val status: String,
        val message: String,
        val data: UserSettingsData?
    )

    data class UserSettingsData(
        val userId: Long?,
        val userType: String?,
        val settings: List<UserSettingValue>?
    )

    data class UserSettingValue(
        val settingKey: String?,
        val value: String?,
        val defaultValue: String?
    )

    data class UpdateSettingResponse(
        val status: String,
        val message: String,
        val data: Any?
    )

    /**
     * Lấy danh sách settings theo type
     * @param settingTypes Danh sách các type cần lấy (AUTO, COMMON, MEDICATION, PRESCRIPTION)
     * @return Result<SettingsByTypeResponse>
     */
    suspend fun getSettingsByType(settingTypes: List<String>): Result<SettingsByTypeResponse> {
        return withContext(Dispatchers.IO) {
            try {
                // Tạo query parameters
                val queryParams = settingTypes.joinToString("&") { "settingTypes=$it" }
                val url = "$baseUrl/type?$queryParams"

                Log.d("UserSettingService", "GetSettingsByType URL: $url")

                val request = Request.Builder()
                    .url(url)
                    .get()
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("UserSettingService", "GetSettingsByType Response Code: ${response.code}")
                Log.d("UserSettingService", "GetSettingsByType Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        val settingsList = mutableListOf<SettingItem>()

                        for (i in 0 until dataArray.length()) {
                            val itemJson = dataArray.getJSONObject(i)
                            settingsList.add(
                                SettingItem(
                                    id = if (itemJson.has("id")) itemJson.getLong("id") else null,
                                    name = itemJson.optString("name", null),
                                    description = itemJson.optString("description", null),
                                    settingKey = itemJson.optString("settingKey", null),
                                    settingType = itemJson.optString("settingType", null),
                                    defaultValue = itemJson.optString("defaultValue", null),
                                    possibleValues = itemJson.optString("possibleValues", null),
                                    isActive = if (itemJson.has("isActive")) itemJson.getBoolean("isActive") else null
                                )
                            )
                        }
                        settingsList
                    } else null

                    Result.success(SettingsByTypeResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể tải danh sách settings")
                        } catch (e: Exception) {
                            "Không thể tải danh sách settings"
                        }
                    } else {
                        "Không thể tải danh sách settings"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("UserSettingService", "GetSettingsByType error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Lấy settings của user hiện tại
     * @param accessToken JWT access token
     * @return Result<UserSettingsResponse>
     */
    suspend fun getUserSettings(accessToken: String): Result<UserSettingsResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/user")
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("UserSettingService", "GetUserSettings Response Code: ${response.code}")
                Log.d("UserSettingService", "GetUserSettings Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    val data = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        val settingsArray = if (dataJson.has("settings") && !dataJson.isNull("settings")) {
                            dataJson.getJSONArray("settings")
                        } else null

                        val settingsList = if (settingsArray != null) {
                            val list = mutableListOf<UserSettingValue>()
                            for (i in 0 until settingsArray.length()) {
                                val settingJson = settingsArray.getJSONObject(i)
                                list.add(
                                    UserSettingValue(
                                        settingKey = settingJson.optString("settingKey", null),
                                        value = settingJson.optString("value", null),
                                        defaultValue = settingJson.optString("defaultValue", null)
                                    )
                                )
                            }
                            list
                        } else null

                        UserSettingsData(
                            userId = if (dataJson.has("userId")) dataJson.getLong("userId") else null,
                            userType = dataJson.optString("userType", null),
                            settings = settingsList
                        )
                    } else null

                    Result.success(UserSettingsResponse(status, message, data))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể tải settings của user")
                        } catch (e: Exception) {
                            "Không thể tải settings của user"
                        }
                    } else {
                        "Không thể tải settings của user"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("UserSettingService", "GetUserSettings error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Cập nhật setting của user
     * @param accessToken JWT access token
     * @param settingKey Key của setting cần update
     * @param value Giá trị mới
     * @return Result<UpdateSettingResponse>
     */
    suspend fun updateUserSetting(
        accessToken: String,
        settingKey: String,
        value: String
    ): Result<UpdateSettingResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("settingKey", settingKey)
                    put("value", value)
                }

                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())

                val request = Request.Builder()
                    .url("$baseUrl/user")
                    .put(requestBody)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("UserSettingService", "UpdateUserSetting Response Code: ${response.code}")
                Log.d("UserSettingService", "UpdateUserSetting Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    Result.success(UpdateSettingResponse(status, message, null))
                } else {
                    val errorMessage = if (responseBody != null) {
                        try {
                            val errorJson = JSONObject(responseBody)
                            errorJson.optString("message", "Không thể cập nhật setting")
                        } catch (e: Exception) {
                            "Không thể cập nhật setting"
                        }
                    } else {
                        "Không thể cập nhật setting"
                    }
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Log.e("UserSettingService", "UpdateUserSetting error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}

