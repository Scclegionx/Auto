package com.auto_fe.auto_fe.service.be

import android.util.Log
import com.auto_fe.auto_fe.models.MedicationLog
import com.auto_fe.auto_fe.models.MedicationLogHistoryResponse
import com.auto_fe.auto_fe.network.ApiClient
import com.auto_fe.auto_fe.config.be.BeConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject

class MedicationLogService {
    private val client by lazy { ApiClient.getClient() }
    private val baseUrl = "${BeConfig.BASE_URL}/medication-logs"

    /**
     * Get my medication history (Elder user)
     * Note: This requires userId - implement proper token parsing or pass userId
     */
    suspend fun getMyMedicationHistory(
        accessToken: String,
        days: Int = 7
    ): Result<MedicationLogHistoryResponse> = withContext(Dispatchers.IO) {
        try {
            // For now, return empty response
            // TODO: Implement proper userId extraction from token or SharedPreferences
            Log.w("MedicationLogService", "getMyMedicationHistory not fully implemented")
            Result.success(MedicationLogHistoryResponse(emptyList(), emptyMap()))
        } catch (e: Exception) {
            Log.e("MedicationLogService", "Error getting my medication history", e)
            Result.failure(e)
        }
    }

    /**
     * Get medication history for a specific elder (Supervisor view)
     */
    suspend fun getElderMedicationHistory(
        accessToken: String,
        elderId: Long,
        days: Int = 7
    ): Result<MedicationLogHistoryResponse> = withContext(Dispatchers.IO) {
        try {
            val url = "$baseUrl/elder/$elderId/history?days=$days"
            Log.d("MedicationLogService", "Requesting URL: $url")
            Log.d("MedicationLogService", "elderId: $elderId, days: $days")

            val request = Request.Builder()
                .url(url)
                .header("Authorization", "Bearer $accessToken")
                .get()
                .build()

            val response = client.newCall(request).execute()
            Log.d("MedicationLogService", "Response code: ${response.code}")

            if (response.isSuccessful) {
                val responseBody = response.body?.string() ?: ""
                Log.d("MedicationLogService", "Response body: $responseBody")

                val jsonResponse = JSONObject(responseBody)
                val status = jsonResponse.getString("status")
                
                if (status == "success") {
                    val dataObj = jsonResponse.getJSONObject("data")
                    val historyResponse = parseMedicationLogHistoryResponse(dataObj)
                    Log.d("MedicationLogService", "Parsed ${historyResponse.logs.size} logs")
                    Result.success(historyResponse)
                } else {
                    val message = jsonResponse.optString("message", "Failed to get medication history")
                    Log.e("MedicationLogService", "API returned error: $message")
                    Result.failure(Exception(message))
                }
            } else {
                val errorBody = response.body?.string() ?: "No error body"
                Log.e("MedicationLogService", "Request failed: ${response.code}, body: $errorBody")
                Result.failure(Exception("Failed to get medication history: ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e("MedicationLogService", "Error getting elder medication history", e)
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Confirm medication taken
     */
    suspend fun confirmMedicationTaken(
        accessToken: String,
        logId: Long,
        userId: Long,
        note: String? = null
    ): Result<MedicationLog> = withContext(Dispatchers.IO) {
        try {
            val url = "$baseUrl/$logId/confirm?userId=$userId"

            val requestBody = if (note != null) {
                JSONObject().apply {
                    put("note", note)
                }.toString().toRequestBody("application/json".toMediaType())
            } else {
                "{}".toRequestBody("application/json".toMediaType())
            }

            val request = Request.Builder()
                .url(url)
                .header("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val responseBody = response.body?.string() ?: ""
                Log.d("MedicationLogService", "Response: $responseBody")

                val jsonResponse = JSONObject(responseBody)
                val status = jsonResponse.getString("status")
                
                if (status == "success") {
                    val dataObj = jsonResponse.getJSONObject("data")
                    val medicationLog = parseMedicationLog(dataObj)
                    Result.success(medicationLog)
                } else {
                    val message = jsonResponse.optString("message", "Failed to confirm medication")
                    Result.failure(Exception(message))
                }
            } else {
                Result.failure(Exception("Failed to confirm medication: ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e("MedicationLogService", "Error confirming medication", e)
            Result.failure(e)
        }
    }

    /**
     * Get medication log detail
     */
    suspend fun getMedicationLogDetail(
        accessToken: String,
        logId: Long
    ): Result<MedicationLog> = withContext(Dispatchers.IO) {
        try {
            val url = "$baseUrl/$logId"

            val request = Request.Builder()
                .url(url)
                .header("Authorization", "Bearer $accessToken")
                .get()
                .build()

            val response = client.newCall(request).execute()

            if (response.isSuccessful) {
                val responseBody = response.body?.string() ?: ""
                Log.d("MedicationLogService", "Response: $responseBody")

                val jsonResponse = JSONObject(responseBody)
                val status = jsonResponse.getString("status")
                
                if (status == "success") {
                    val dataObj = jsonResponse.getJSONObject("data")
                    val medicationLog = parseMedicationLog(dataObj)
                    Result.success(medicationLog)
                } else {
                    val message = jsonResponse.optString("message", "Failed to get medication log")
                    Result.failure(Exception(message))
                }
            } else {
                Result.failure(Exception("Failed to get medication log: ${response.code}"))
            }
        } catch (e: Exception) {
            Log.e("MedicationLogService", "Error getting medication log detail", e)
            Result.failure(e)
        }
    }

    // Helper functions to parse JSON
    private fun parseMedicationLogHistoryResponse(json: JSONObject): MedicationLogHistoryResponse {
        val logsArray = json.getJSONArray("logs")
        val logs = mutableListOf<MedicationLog>()
        
        for (i in 0 until logsArray.length()) {
            logs.add(parseMedicationLog(logsArray.getJSONObject(i)))
        }
        
        val statisticsObj = json.getJSONObject("statistics")
        val statistics = mapOf(
            "total" to statisticsObj.optInt("total", 0),
            "taken" to statisticsObj.optInt("taken", 0),
            "missed" to statisticsObj.optInt("missed", 0),
            "onTime" to statisticsObj.optInt("onTime", 0),
            "adherenceRate" to statisticsObj.optDouble("adherenceRate", 0.0),
            "onTimeRate" to statisticsObj.optDouble("onTimeRate", 0.0)
        )
        
        return MedicationLogHistoryResponse(logs, statistics)
    }

    private fun parseMedicationLog(json: JSONObject): MedicationLog {
        return MedicationLog(
            id = json.getLong("id"),
            elderUserId = json.getLong("elderUserId"),
            medicationIds = json.getString("medicationIds"),
            medicationNames = json.getString("medicationNames"),
            medicationCount = json.getInt("medicationCount"),
            reminderTime = json.getString("reminderTime"),
            actualTakenTime = json.optString("actualTakenTime").takeIf { it.isNotBlank() },
            status = json.getString("status"),
            minutesLate = json.optInt("minutesLate").takeIf { json.has("minutesLate") },
            note = json.optString("note").takeIf { it.isNotBlank() },
            fcmSent = json.optBoolean("fcmSent", false),
            fcmSentTime = json.optString("fcmSentTime").takeIf { it.isNotBlank() },
            createdAt = json.getString("createdAt"),
            updatedAt = json.getString("updatedAt")
        )
    }
}
