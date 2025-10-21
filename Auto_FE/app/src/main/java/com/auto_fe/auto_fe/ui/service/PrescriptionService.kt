package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.ui.screens.MedicationReminderForm
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Service để xử lý Prescription (Đơn thuốc) với Auto_BE API
 */
class PrescriptionService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val baseUrl = "http://192.168.33.103:8080/api/cron-prescriptions"

    /**
     * Data classes
     */
    data class PrescriptionListResponse(
        val status: String,
        val message: String,
        val data: List<Prescription>?
    )

    data class PrescriptionDetailResponse(
        val status: String,
        val message: String,
        val data: Prescription?
    )

    data class Prescription(
        val id: Long,
        val name: String,
        val description: String?,
        val imageUrl: String?,
        val isActive: Boolean,
        val userId: Long,
        val medicationReminders: List<MedicationReminder>
    )

    data class MedicationReminder(
        val id: Long,
        val name: String,
        val description: String?,
        val type: String, // "BEFORE_MEAL", "AFTER_MEAL", "WITH_MEAL"
        val reminderTime: String, // "HH:mm"
        val daysOfWeek: String, // "1111111" = everyday
        val isActive: Boolean,
        val prescriptionId: Long,
        val userId: Long
    )

    /**
     * Lấy danh sách tất cả đơn thuốc của user
     */
    suspend fun getAllPrescriptions(accessToken: String): Result<PrescriptionListResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(baseUrl)
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("PrescriptionService", "Get All Response Code: ${response.code}")
                Log.d("PrescriptionService", "Get All Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    val prescriptions = mutableListOf<Prescription>()
                    if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        for (i in 0 until dataArray.length()) {
                            val prescriptionJson = dataArray.getJSONObject(i)
                            prescriptions.add(parsePrescription(prescriptionJson))
                        }
                    }

                    Result.success(PrescriptionListResponse(status, message, prescriptions))
                } else {
                    Result.failure(Exception("Failed to get prescriptions: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("PrescriptionService", "Get all prescriptions error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Lấy chi tiết đơn thuốc theo ID
     */
    suspend fun getPrescriptionById(prescriptionId: Long, accessToken: String): Result<PrescriptionDetailResponse> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/$prescriptionId")
                    .get()
                    .addHeader("Authorization", "Bearer $accessToken")
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                Log.d("PrescriptionService", "Get By ID Response Code: ${response.code}")
                Log.d("PrescriptionService", "Get By ID Response Body: $responseBody")

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")

                    val prescription = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        parsePrescription(dataJson)
                    } else null

                    Result.success(PrescriptionDetailResponse(status, message, prescription))
                } else {
                    Result.failure(Exception("Failed to get prescription: ${response.code}"))
                }
            } catch (e: Exception) {
                Log.e("PrescriptionService", "Get prescription by id error", e)
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Parse JSON thành Prescription object
     */
    private fun parsePrescription(json: JSONObject): Prescription {
        val medications = mutableListOf<MedicationReminder>()
        if (json.has("medicationReminders") && !json.isNull("medicationReminders")) {
            val medicationsArray = json.getJSONArray("medicationReminders")
            for (i in 0 until medicationsArray.length()) {
                val medJson = medicationsArray.getJSONObject(i)
                medications.add(
                    MedicationReminder(
                        id = medJson.getLong("id"),
                        name = medJson.getString("name"),
                        description = medJson.optString("description"),
                        type = medJson.optString("type", "AFTER_MEAL"),
                        reminderTime = medJson.getString("reminderTime"),
                        daysOfWeek = medJson.optString("daysOfWeek", "1111111"),
                        isActive = medJson.optBoolean("isActive", true),
                        prescriptionId = medJson.getLong("prescriptionId"),
                        userId = medJson.getLong("userId")
                    )
                )
            }
        }

        return Prescription(
            id = json.getLong("id"),
            name = json.getString("name"),
            description = json.optString("description"),
            imageUrl = json.optString("imageUrl"),
            isActive = json.optBoolean("isActive", true),
            userId = json.getLong("userId"),
            medicationReminders = medications
        )
    }

    /**
     * Tạo đơn thuốc mới
     */
    suspend fun createPrescription(
        name: String,
        description: String,
        imageUrl: String,
        medications: List<MedicationReminderForm>,
        accessToken: String
    ): Result<PrescriptionDetailResponse> = withContext(Dispatchers.IO) {
        try {
            Log.d("PrescriptionService", "Creating prescription: $name")

            // Build request body
            val requestBody = JSONObject().apply {
                put("name", name)
                put("description", description)
                put("imageUrl", imageUrl)

                val medicationsArray = JSONArray()
                medications.forEach { med ->
                    val medJson = JSONObject().apply {
                        put("name", med.name)
                        
                        // Ghép meal timing vào description
                        val mealTimingText = when (med.mealTiming) {
                            "BEFORE_MEAL" -> "Uống trước ăn"
                            "AFTER_MEAL" -> "Uống sau ăn"
                            "WITH_MEAL" -> "Uống trong bữa ăn"
                            else -> ""
                        }
                        val fullDescription = if (med.description.isNotBlank()) {
                            "${med.description}\n📝 $mealTimingText"
                        } else {
                            "📝 $mealTimingText"
                        }
                        put("description", fullDescription)
                        
                        put("type", med.type) // PRESCRIPTION or OVER_THE_COUNTER

                        // Convert List<String> to JSONArray
                        val timesArray = JSONArray()
                        med.reminderTimes.forEach { time ->
                            timesArray.put(time)
                        }
                        put("reminderTimes", timesArray)

                        put("daysOfWeek", med.daysOfWeek)
                    }
                    medicationsArray.put(medJson)
                }
                put("medicationReminders", medicationsArray)
            }

            Log.d("PrescriptionService", "Request body: ${requestBody.toString(2)}")

            val request = Request.Builder()
                .url("$baseUrl/create")
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("Content-Type", "application/json")
                .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()

            Log.d("PrescriptionService", "Response code: ${response.code}")
            Log.d("PrescriptionService", "Response body: $responseBody")

            if (response.isSuccessful && responseBody != null) {
                val json = JSONObject(responseBody)
                val prescription = parsePrescription(json.getJSONObject("data"))

                Result.success(
                    PrescriptionDetailResponse(
                        status = json.getString("status"),
                        message = json.getString("message"),
                        data = prescription
                    )
                )
            } else {
                val errorMessage = if (responseBody != null) {
                    try {
                        val json = JSONObject(responseBody)
                        json.optString("message", "Failed to create prescription")
                    } catch (e: Exception) {
                        responseBody
                    }
                } else {
                    "Unknown error"
                }
                Log.e("PrescriptionService", "Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in createPrescription", e)
            Result.failure(e)
        }
    }
}
