package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.ui.screens.MedicationReminderForm
import com.auto_fe.auto_fe.network.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Service để xử lý Prescription (Đơn thuốc) với Auto_BE API
 */
class PrescriptionService {
    private val client by lazy { ApiClient.getClient() }

    private val baseUrl = com.auto_fe.auto_fe.network.ApiConfig.BASE_URL + "/cron-prescriptions"

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
        val medications: List<Medication>? = null,  // ✅ New grouped format
        val medicationReminders: List<MedicationReminder>? = null  // Legacy
    )

    // ✅ New grouped medication format
    data class Medication(
        val id: Long,
        val medicationName: String,
        val notes: String?,
        val type: String,
        val reminderTimes: List<String>,  // ["08:00", "12:00", "18:00"]
        val daysOfWeek: String,
        val isActive: Boolean,
        val prescriptionId: Long,
        val userId: Long
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
                    Result.failure(Exception("Không thể tải danh sách đơn thuốc: ${response.code}"))
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
                    Result.failure(Exception("Không thể tải chi tiết đơn thuốc: ${response.code}"))
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
        // ✅ Parse medications (grouped format)
        val medications = mutableListOf<Medication>()
        if (json.has("medications") && !json.isNull("medications")) {
            val medicationsArray = json.getJSONArray("medications")
            for (i in 0 until medicationsArray.length()) {
                val medJson = medicationsArray.getJSONObject(i)
                
                // Parse reminderTimes array
                val reminderTimes = mutableListOf<String>()
                if (medJson.has("reminderTimes") && !medJson.isNull("reminderTimes")) {
                    val timesArray = medJson.getJSONArray("reminderTimes")
                    for (j in 0 until timesArray.length()) {
                        reminderTimes.add(timesArray.getString(j))
                    }
                }
                
                medications.add(
                    Medication(
                        id = medJson.getLong("id"),
                        medicationName = medJson.getString("medicationName"),
                        notes = medJson.optString("notes"),
                        type = medJson.optString("type", "PRESCRIPTION"),
                        reminderTimes = reminderTimes,
                        daysOfWeek = medJson.optString("daysOfWeek", "1111111"),
                        isActive = medJson.optBoolean("isActive", true),
                        prescriptionId = medJson.getLong("prescriptionId"),
                        userId = medJson.getLong("userId")
                    )
                )
            }
        }
        
        // Legacy: Parse medicationReminders (for backward compatibility)
        val medicationReminders = mutableListOf<MedicationReminder>()
        if (json.has("medicationReminders") && !json.isNull("medicationReminders")) {
            val medicationsArray = json.getJSONArray("medicationReminders")
            for (i in 0 until medicationsArray.length()) {
                val medJson = medicationsArray.getJSONObject(i)
                medicationReminders.add(
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
            medications = medications.takeIf { it.isNotEmpty() },  // ✅ New field
            medicationReminders = medicationReminders.takeIf { it.isNotEmpty() }  // Legacy
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
                        
                        // Ghi chú thuốc (user tự nhập, bao gồm cả thời điểm uống nếu muốn)
                        put("description", med.description)
                        
                        put("type", med.type) // Mặc định là PRESCRIPTION

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
                        json.optString("message", "Không thể tạo đơn thuốc")
                    } catch (e: Exception) {
                        "Lỗi không xác định"
                    }
                } else {
                    "Lỗi không xác định"
                }
                Log.e("PrescriptionService", "Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in createPrescription", e)
            Result.failure(Exception("Lỗi kết nối: ${e.message}"))
        }
    }

    /**
     * Tạo đơn thuốc mới với upload ảnh
     * ✅ Lazy Upload: Chỉ upload khi user submit form
     * ✅ Atomic operation: Upload + Create DB cùng lúc
     * ✅ Không có orphan files
     */
    suspend fun createPrescriptionWithImage(
        name: String,
        description: String,
        imageFile: File,
        medications: List<MedicationReminderForm>,
        accessToken: String
    ): Result<PrescriptionDetailResponse> = withContext(Dispatchers.IO) {
        try {
            Log.d("PrescriptionService", "Creating prescription with image: $name")

            // Build prescription data JSON
            val prescriptionData = JSONObject().apply {
                put("name", name)
                put("description", description)
                put("imageUrl", "") // Backend sẽ set sau khi upload

                val medicationsArray = JSONArray()
                medications.forEach { med ->
                    val medJson = JSONObject().apply {
                        put("name", med.name)
                        put("description", med.description)
                        put("type", med.type)

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

            Log.d("PrescriptionService", "Prescription data: ${prescriptionData.toString(2)}")

            // Detect MIME type from file extension
            val extension = imageFile.extension.lowercase()
            val mimeType = when (extension) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                "webp" -> "image/webp"
                "gif" -> "image/gif"
                else -> "image/jpeg"
            }

            // Build multipart request
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("data", prescriptionData.toString())
                .addFormDataPart(
                    "image",
                    imageFile.name,
                    imageFile.asRequestBody(mimeType.toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url("$baseUrl/create-with-image")
                .addHeader("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            Log.d("PrescriptionService", "Sending multipart request to /create-with-image")

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
                        json.optString("message", "Không thể tạo đơn thuốc")
                    } catch (e: Exception) {
                        "Lỗi không xác định"
                    }
                } else {
                    "Lỗi không xác định"
                }
                Log.e("PrescriptionService", "Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in createPrescriptionWithImage", e)
            Result.failure(Exception("Lỗi kết nối: ${e.message}"))
        }
    }

    /**
     * Cập nhật đơn thuốc
     */
    suspend fun updatePrescription(
        prescriptionId: Long,
        name: String,
        description: String,
        imageUrl: String,
        medications: List<MedicationReminderForm>,
        accessToken: String
    ): Result<PrescriptionDetailResponse> = withContext(Dispatchers.IO) {
        try {
            Log.d("PrescriptionService", "Updating prescription ID: $prescriptionId")
            Log.d("PrescriptionService", "Name: $name, Medications: ${medications.size}")

            val jsonBody = JSONObject().apply {
                put("name", name)
                put("description", description)
                put("imageUrl", imageUrl)

                val medicationsArray = JSONArray()
                medications.forEach { med ->
                    val medJson = JSONObject().apply {
                        put("name", med.name)
                        put("description", med.description)
                        put("type", med.type)
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

            Log.d("PrescriptionService", "Update Request Body: $jsonBody")

            val requestBody = jsonBody.toString()
                .toRequestBody("application/json; charset=utf-8".toMediaType())

            val request = Request.Builder()
                .url("$baseUrl/$prescriptionId")
                .put(requestBody)
                .addHeader("Authorization", "Bearer $accessToken")
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()

            Log.d("PrescriptionService", "Update Response Code: ${response.code}")
            Log.d("PrescriptionService", "Update Response Body: $responseBody")

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
                        json.optString("message", "Không thể cập nhật đơn thuốc")
                    } catch (e: Exception) {
                        "Lỗi không xác định"
                    }
                } else {
                    "Lỗi không xác định"
                }
                Log.e("PrescriptionService", "Update Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in updatePrescription", e)
            Result.failure(Exception("Lỗi kết nối: ${e.message}"))
        }
    }

    /**
     * Xóa đơn thuốc
     */
    suspend fun deletePrescription(
        prescriptionId: Long,
        accessToken: String
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            Log.d("PrescriptionService", "Deleting prescription: $prescriptionId")

            val request = Request.Builder()
                .url("$baseUrl/$prescriptionId")
                .delete()
                .addHeader("Authorization", "Bearer $accessToken")
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()

            Log.d("PrescriptionService", "Delete Response Code: ${response.code}")
            Log.d("PrescriptionService", "Delete Response Body: $responseBody")

            if (response.isSuccessful && responseBody != null) {
                val json = JSONObject(responseBody)
                val message = json.optString("message", "Đã xóa đơn thuốc thành công")
                Result.success(message)
            } else {
                val errorMessage = if (responseBody != null) {
                    try {
                        val json = JSONObject(responseBody)
                        json.optString("message", "Không thể xóa đơn thuốc")
                    } catch (e: Exception) {
                        "Lỗi không xác định"
                    }
                } else {
                    "Lỗi không xác định"
                }
                Log.e("PrescriptionService", "Delete Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in deletePrescription", e)
            Result.failure(Exception("Lỗi kết nối: ${e.message}"))
        }
    }

    /**
     * Toggle trạng thái đơn thuốc (active/inactive)
     */
    suspend fun togglePrescriptionStatus(
        prescriptionId: Long,
        accessToken: String
    ): Result<PrescriptionDetailResponse> = withContext(Dispatchers.IO) {
        try {
            Log.d("PrescriptionService", "Toggling prescription status: $prescriptionId")

            val emptyBody = ByteArray(0)
            val request = Request.Builder()
                .url("$baseUrl/$prescriptionId/toggle-status")
                .patch(emptyBody.toRequestBody(null))
                .addHeader("Authorization", "Bearer $accessToken")
                .addHeader("Content-Type", "application/json")
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()

            Log.d("PrescriptionService", "Toggle Status Response Code: ${response.code}")
            Log.d("PrescriptionService", "Toggle Status Response Body: $responseBody")

            if (response.isSuccessful && responseBody != null) {
                val json = JSONObject(responseBody)
                val dataJson = json.getJSONObject("data")
                
                val prescription = parsePrescription(dataJson)
                Result.success(PrescriptionDetailResponse(
                    status = "success",
                    message = json.optString("message", "Đã cập nhật trạng thái"),
                    data = prescription
                ))
            } else {
                val errorMessage = if (responseBody != null) {
                    try {
                        val json = JSONObject(responseBody)
                        json.optString("message", "Không thể cập nhật trạng thái")
                    } catch (e: Exception) {
                        "Lỗi không xác định"
                    }
                } else {
                    "Lỗi không xác định"
                }
                Log.e("PrescriptionService", "Toggle Status Error: $errorMessage")
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Log.e("PrescriptionService", "Exception in togglePrescriptionStatus", e)
            Result.failure(Exception("Lỗi kết nối: ${e.message}"))
        }
    }
}
