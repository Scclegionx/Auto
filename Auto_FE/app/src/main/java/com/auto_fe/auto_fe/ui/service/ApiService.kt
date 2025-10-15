package com.auto_fe.auto_fe.ui.service

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Service layer để gọi API Backend
 * Tách biệt logic gọi API khỏi UI components
 */
class ApiService(private val context: Context) {
    
    companion object {
        private const val TAG = "ApiService"
        private const val BASE_URL = "https://your-backend-api.com/api" // Thay đổi URL thực tế
        private const val TIMEOUT = 10000 // 10 seconds
    }

    /**
     * Gửi dữ liệu giọng nói lên server để xử lý
     */
    suspend fun sendVoiceData(
        audioData: ByteArray,
        transcript: String,
        userId: String? = null
    ): ApiResult<VoiceResponse> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Sending voice data to server...")
            
            val url = URL("$BASE_URL/voice/process")
            val connection = url.openConnection() as HttpURLConnection
            
            connection.apply {
                requestMethod = "POST"
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Accept", "application/json")
                connectTimeout = TIMEOUT
                readTimeout = TIMEOUT
            }
            
            val requestBody = JSONObject().apply {
                put("transcript", transcript)
                put("audioData", android.util.Base64.encodeToString(audioData, android.util.Base64.DEFAULT))
                put("userId", userId)
                put("timestamp", System.currentTimeMillis())
            }
            
            connection.outputStream.use { outputStream ->
                outputStream.write(requestBody.toString().toByteArray())
            }
            
            val responseCode = connection.responseCode
            val responseBody = connection.inputStream.bufferedReader().readText()
            
            Log.d(TAG, "Response code: $responseCode")
            Log.d(TAG, "Response body: $responseBody")
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val jsonResponse = JSONObject(responseBody)
                val voiceResponse = VoiceResponse(
                    success = jsonResponse.optBoolean("success", false),
                    message = jsonResponse.optString("message", ""),
                    action = jsonResponse.optString("action", ""),
                    data = jsonResponse.optJSONObject("data")
                )
                ApiResult.Success(voiceResponse)
            } else {
                ApiResult.Error("Server error: $responseCode")
            }
            
        } catch (e: IOException) {
            Log.e(TAG, "Network error: ${e.message}")
            ApiResult.Error("Network error: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error: ${e.message}")
            ApiResult.Error("Unexpected error: ${e.message}")
        }
    }

    /**
     * Lấy danh sách thuốc của user
     */
    suspend fun getMedicines(userId: String? = null): ApiResult<List<MedicineData>> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Fetching medicines for user: $userId")
            
            val url = URL("$BASE_URL/medicines?userId=${userId ?: "default"}")
            val connection = url.openConnection() as HttpURLConnection
            
            connection.apply {
                requestMethod = "GET"
                setRequestProperty("Accept", "application/json")
                connectTimeout = TIMEOUT
                readTimeout = TIMEOUT
            }
            
            val responseCode = connection.responseCode
            val responseBody = connection.inputStream.bufferedReader().readText()
            
            Log.d(TAG, "Medicines response code: $responseCode")
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val jsonResponse = JSONObject(responseBody)
                val medicinesArray = jsonResponse.optJSONArray("medicines")
                
                val medicines = mutableListOf<MedicineData>()
                if (medicinesArray != null) {
                    for (i in 0 until medicinesArray.length()) {
                        val medicineJson = medicinesArray.getJSONObject(i)
                        medicines.add(
                            MedicineData(
                                id = medicineJson.optString("id"),
                                name = medicineJson.optString("name"),
                                dosage = medicineJson.optString("dosage"),
                                time = medicineJson.optString("time"),
                                status = medicineJson.optString("status")
                            )
                        )
                    }
                }
                
                ApiResult.Success(medicines)
            } else {
                ApiResult.Error("Failed to fetch medicines: $responseCode")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching medicines: ${e.message}")
            ApiResult.Error("Error fetching medicines: ${e.message}")
        }
    }

    /**
     * Cập nhật cài đặt user
     */
    suspend fun updateSettings(
        userId: String?,
        settings: UserSettings
    ): ApiResult<Boolean> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Updating settings for user: $userId")
            
            val url = URL("$BASE_URL/settings")
            val connection = url.openConnection() as HttpURLConnection
            
            connection.apply {
                requestMethod = "PUT"
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Accept", "application/json")
                connectTimeout = TIMEOUT
                readTimeout = TIMEOUT
            }
            
            val requestBody = JSONObject().apply {
                put("userId", userId)
                put("voiceEnabled", settings.voiceEnabled)
                put("notificationEnabled", settings.notificationEnabled)
                put("autoStartEnabled", settings.autoStartEnabled)
                put("language", settings.language)
                put("theme", settings.theme)
            }
            
            connection.outputStream.use { outputStream ->
                outputStream.write(requestBody.toString().toByteArray())
            }
            
            val responseCode = connection.responseCode
            val responseBody = connection.inputStream.bufferedReader().readText()
            
            Log.d(TAG, "Settings update response code: $responseCode")
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val jsonResponse = JSONObject(responseBody)
                val success = jsonResponse.optBoolean("success", false)
                ApiResult.Success(success)
            } else {
                ApiResult.Error("Failed to update settings: $responseCode")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error updating settings: ${e.message}")
            ApiResult.Error("Error updating settings: ${e.message}")
        }
    }

    /**
     * Gửi feedback từ user
     */
    suspend fun sendFeedback(
        userId: String?,
        feedback: String,
        rating: Int
    ): ApiResult<Boolean> = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Sending feedback from user: $userId")
            
            val url = URL("$BASE_URL/feedback")
            val connection = url.openConnection() as HttpURLConnection
            
            connection.apply {
                requestMethod = "POST"
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Accept", "application/json")
                connectTimeout = TIMEOUT
                readTimeout = TIMEOUT
            }
            
            val requestBody = JSONObject().apply {
                put("userId", userId)
                put("feedback", feedback)
                put("rating", rating)
                put("timestamp", System.currentTimeMillis())
            }
            
            connection.outputStream.use { outputStream ->
                outputStream.write(requestBody.toString().toByteArray())
            }
            
            val responseCode = connection.responseCode
            val responseBody = connection.inputStream.bufferedReader().readText()
            
            Log.d(TAG, "Feedback response code: $responseCode")
            
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val jsonResponse = JSONObject(responseBody)
                val success = jsonResponse.optBoolean("success", false)
                ApiResult.Success(success)
            } else {
                ApiResult.Error("Failed to send feedback: $responseCode")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error sending feedback: ${e.message}")
            ApiResult.Error("Error sending feedback: ${e.message}")
        }
    }
}

/**
 * Data classes cho API responses
 */
data class VoiceResponse(
    val success: Boolean,
    val message: String,
    val action: String,
    val data: JSONObject?
)

data class MedicineData(
    val id: String,
    val name: String,
    val dosage: String,
    val time: String,
    val status: String
)

data class UserSettings(
    val voiceEnabled: Boolean,
    val notificationEnabled: Boolean,
    val autoStartEnabled: Boolean,
    val language: String,
    val theme: String
)

/**
 * Sealed class cho API results
 */
sealed class ApiResult<out T> {
    data class Success<T>(val data: T) : ApiResult<T>()
    data class Error(val message: String) : ApiResult<Nothing>()
}
