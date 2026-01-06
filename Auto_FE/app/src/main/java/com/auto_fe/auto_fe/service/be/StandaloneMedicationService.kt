package com.auto_fe.auto_fe.service.be

import com.auto_fe.auto_fe.config.be.BeConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class StandaloneMedicationService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val baseUrl = "${BeConfig.BASE_URL}/medication"
    private val mediaType = "application/json; charset=utf-8".toMediaType()

    /**
     * Data classes
     */
    data class Medication(
        val id: Long,
        val name: String,
        val dosage: String?,
        val unit: String?,
        val frequency: String?,
        val times: List<String>,
        val daysOfWeek: String?,        // "1111111" = everyday
        val duration: Int?,
        val description: String?,       // Mô tả/ghi chú
        val isActive: Boolean,
        val userId: Long,
        val createdAt: String,
        val updatedAt: String
    )

    data class MedicationRequest(
        val name: String,
        val description: String?,
        val type: String = "OVER_THE_COUNTER",
        val reminderTimes: List<String>,
        val daysOfWeek: String,
        val isActive: Boolean = true,
        val elderUserId: Long? = null  // For Supervisor creating medication for Elder
    )

    data class BaseResponse<T>(
        val status: String,
        val message: String,
        val data: T?
    )

    /**
     * Lấy danh sách medication (sử dụng authentication từ token)
     * Gọi endpoint standalone để lấy chỉ thuốc ngoài đơn đã được group
     */
    suspend fun getAll(accessToken: String, userId: Long): Result<BaseResponse<List<Medication>>> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/standalone/user/$userId")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .get()
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val medications = mutableListOf<Medication>()
                    
                    if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        for (i in 0 until dataArray.length()) {
                            val medicationJson = dataArray.getJSONObject(i)
                            
                            // Parse reminderTimes array
                            val timesArray = medicationJson.getJSONArray("reminderTimes")
                            val times = mutableListOf<String>()
                            for (j in 0 until timesArray.length()) {
                                times.add(timesArray.getString(j))
                            }
                            
                            medications.add(
                                Medication(
                                    id = medicationJson.getLong("id"),
                                    name = medicationJson.getString("medicationName"),
                                    dosage = null, // Backend không có field này trong response
                                    unit = null,   // Backend không có field này trong response
                                    frequency = null, // Backend không có field này trong response
                                    times = times,
                                    daysOfWeek = if (medicationJson.isNull("daysOfWeek")) null else medicationJson.getString("daysOfWeek"),
                                    duration = null, // Backend không có field này trong response
                                    description = if (medicationJson.isNull("description")) null else medicationJson.getString("description"),
                                    isActive = medicationJson.getBoolean("isActive"),
                                    userId = medicationJson.getLong("userId"),
                                    createdAt = medicationJson.getString("createdAt"),
                                    updatedAt = medicationJson.getString("updatedAt")
                                )
                            )
                        }
                    }
                    
                    Result.success(BaseResponse(status, message, medications))
                } else {
                    val errorMessage = responseBody?.let {
                        try {
                            JSONObject(it).optString("message", "Lỗi không xác định")
                        } catch (e: Exception) {
                            "Lỗi không xác định"
                        }
                    } ?: "Lỗi không xác định"
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Tạo thuốc ngoài đơn mới
     */
    suspend fun create(
        accessToken: String,
        request: MedicationRequest
    ): Result<BaseResponse<List<Medication>>> {
        return withContext(Dispatchers.IO) {
            try {
                val reminderTimesArray = JSONArray()
                request.reminderTimes.forEach { reminderTimesArray.put(it) }
                
                val jsonBody = JSONObject().apply {
                    put("name", request.name)
                    put("description", request.description ?: JSONObject.NULL)
                    put("type", request.type)
                    put("reminderTimes", reminderTimesArray)
                    put("daysOfWeek", request.daysOfWeek)
                    put("isActive", request.isActive)
                    put("prescriptionId", JSONObject.NULL) // Standalone medication
                    
                    // Include elderUserId if Supervisor is creating for Elder
                    if (request.elderUserId != null) {
                        put("elderUserId", request.elderUserId)
                    }
                }

                val httpRequest = Request.Builder()
                    .url(baseUrl)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .addHeader("Content-Type", "application/json")
                    .post(jsonBody.toString().toRequestBody(mediaType))
                    .build()

                val response = client.newCall(httpRequest).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val medications = mutableListOf<Medication>()
                    
                    if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        for (i in 0 until dataArray.length()) {
                            val dataJson = dataArray.getJSONObject(i)
                            
                            val timesArray = dataJson.getJSONArray("reminderTimes")
                            val times = mutableListOf<String>()
                            for (j in 0 until timesArray.length()) {
                                times.add(timesArray.getString(j))
                            }
                            
                            medications.add(
                                Medication(
                                    id = dataJson.getLong("id"),
                                    name = dataJson.getString("medicationName"),
                                    dosage = null,
                                    unit = null,
                                    frequency = null,
                                    times = times,
                                    daysOfWeek = if (dataJson.isNull("daysOfWeek")) null else dataJson.getString("daysOfWeek"),
                                    duration = null,
                                    description = if (dataJson.isNull("description")) null else dataJson.getString("description"),
                                    isActive = dataJson.getBoolean("isActive"),
                                    userId = dataJson.getLong("userId"),
                                    createdAt = dataJson.getString("createdAt"),
                                    updatedAt = dataJson.getString("updatedAt")
                                )
                            )
                        }
                    }
                    
                    Result.success(BaseResponse(status, message, medications))
                } else {
                    val errorMessage = responseBody?.let {
                        try {
                            JSONObject(it).optString("message", "Lỗi không xác định")
                        } catch (e: Exception) {
                            "Lỗi không xác định"
                        }
                    } ?: "Lỗi không xác định"
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Xóa thuốc ngoài đơn
     */
    suspend fun delete(accessToken: String, id: Long): Result<BaseResponse<Unit>> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/$id")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .delete()
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(BaseResponse(status, message, null))
                } else {
                    val errorMessage = responseBody?.let {
                        try {
                            JSONObject(it).optString("message", "Lỗi không xác định")
                        } catch (e: Exception) {
                            "Lỗi không xác định"
                        }
                    } ?: "Lỗi không xác định"
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }

    /**
     * Toggle trạng thái active/inactive
     */
    suspend fun toggleActive(accessToken: String, id: Long): Result<BaseResponse<Medication>> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/$id/toggle")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .put("".toRequestBody(mediaType))
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val medication = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        
                        val timesArray = dataJson.getJSONArray("reminderTimes")
                        val times = mutableListOf<String>()
                        for (j in 0 until timesArray.length()) {
                            times.add(timesArray.getString(j))
                        }
                        
                        Medication(
                            id = dataJson.getLong("id"),
                            name = dataJson.getString("medicationName"),
                            dosage = null,
                            unit = null,
                            frequency = null,
                            times = times,
                            daysOfWeek = if (dataJson.isNull("daysOfWeek")) null else dataJson.getString("daysOfWeek"),
                            duration = null,
                            description = if (dataJson.isNull("description")) null else dataJson.getString("description"),
                            isActive = dataJson.getBoolean("isActive"),
                            userId = dataJson.getLong("userId"),
                            createdAt = dataJson.getString("createdAt"),
                            updatedAt = dataJson.getString("updatedAt")
                        )
                    } else null
                    
                    Result.success(BaseResponse(status, message, medication))
                } else {
                    val errorMessage = responseBody?.let {
                        try {
                            JSONObject(it).optString("message", "Lỗi không xác định")
                        } catch (e: Exception) {
                            "Lỗi không xác định"
                        }
                    } ?: "Lỗi không xác định"
                    Result.failure(Exception(errorMessage))
                }
            } catch (e: Exception) {
                Result.failure(Exception("Lỗi kết nối: ${e.message}"))
            }
        }
    }
}
