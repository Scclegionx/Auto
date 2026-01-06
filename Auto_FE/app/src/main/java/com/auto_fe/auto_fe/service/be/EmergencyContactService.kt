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

class EmergencyContactService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val baseUrl = "${BeConfig.BASE_URL}/emergency-contacts"
    private val mediaType = "application/json; charset=utf-8".toMediaType()

    /**
     * Data classes
     */
    data class EmergencyContact(
        val id: Long,
        val name: String,
        val phoneNumber: String,
        val address: String?,
        val relationship: String,
        val note: String?,
        val userId: Long,
        val createdAt: String,
        val updatedAt: String
    )

    data class EmergencyContactRequest(
        val name: String,
        val phoneNumber: String,
        val address: String?,
        val relationship: String,
        val note: String?
    )

    data class BaseResponse<T>(
        val status: String,
        val message: String,
        val data: T?
    )

    /**
     * Lấy danh sách liên hệ khẩn cấp
     */
    suspend fun getAll(accessToken: String): Result<BaseResponse<List<EmergencyContact>>> {
        return withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url(baseUrl)
                    .addHeader("Authorization", "Bearer $accessToken")
                    .get()
                    .build()

                val response = client.newCall(request).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val contacts = mutableListOf<EmergencyContact>()
                    if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataArray = jsonResponse.getJSONArray("data")
                        for (i in 0 until dataArray.length()) {
                            val contactJson = dataArray.getJSONObject(i)
                            contacts.add(
                                EmergencyContact(
                                    id = contactJson.getLong("id"),
                                    name = contactJson.getString("name"),
                                    phoneNumber = contactJson.getString("phoneNumber"),
                                    address = if (contactJson.isNull("address")) null else contactJson.getString("address"),
                                    relationship = contactJson.getString("relationship"),
                                    note = if (contactJson.isNull("note")) null else contactJson.getString("note"),
                                    userId = contactJson.getLong("userId"),
                                    createdAt = contactJson.getString("createdAt"),
                                    updatedAt = contactJson.getString("updatedAt")
                                )
                            )
                        }
                    }
                    
                    Result.success(BaseResponse(status, message, contacts))
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
     * Tạo liên hệ khẩn cấp mới
     */
    suspend fun create(
        accessToken: String,
        request: EmergencyContactRequest
    ): Result<BaseResponse<EmergencyContact>> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("name", request.name)
                    put("phoneNumber", request.phoneNumber)
                    put("address", request.address ?: JSONObject.NULL)
                    put("relationship", request.relationship)
                    put("note", request.note ?: JSONObject.NULL)
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
                    
                    val contact = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        EmergencyContact(
                            id = dataJson.getLong("id"),
                            name = dataJson.getString("name"),
                            phoneNumber = dataJson.getString("phoneNumber"),
                            address = if (dataJson.isNull("address")) null else dataJson.getString("address"),
                            relationship = dataJson.getString("relationship"),
                            note = if (dataJson.isNull("note")) null else dataJson.getString("note"),
                            userId = dataJson.getLong("userId"),
                            createdAt = dataJson.getString("createdAt"),
                            updatedAt = dataJson.getString("updatedAt")
                        )
                    } else null
                    
                    Result.success(BaseResponse(status, message, contact))
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
     * Cập nhật liên hệ khẩn cấp
     */
    suspend fun update(
        accessToken: String,
        id: Long,
        request: EmergencyContactRequest
    ): Result<BaseResponse<EmergencyContact>> {
        return withContext(Dispatchers.IO) {
            try {
                val jsonBody = JSONObject().apply {
                    put("name", request.name)
                    put("phoneNumber", request.phoneNumber)
                    put("address", request.address ?: JSONObject.NULL)
                    put("relationship", request.relationship)
                    put("note", request.note ?: JSONObject.NULL)
                }

                val httpRequest = Request.Builder()
                    .url("$baseUrl/$id")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .addHeader("Content-Type", "application/json")
                    .put(jsonBody.toString().toRequestBody(mediaType))
                    .build()

                val response = client.newCall(httpRequest).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    val contact = if (jsonResponse.has("data") && !jsonResponse.isNull("data")) {
                        val dataJson = jsonResponse.getJSONObject("data")
                        EmergencyContact(
                            id = dataJson.getLong("id"),
                            name = dataJson.getString("name"),
                            phoneNumber = dataJson.getString("phoneNumber"),
                            address = if (dataJson.isNull("address")) null else dataJson.getString("address"),
                            relationship = dataJson.getString("relationship"),
                            note = if (dataJson.isNull("note")) null else dataJson.getString("note"),
                            userId = dataJson.getLong("userId"),
                            createdAt = dataJson.getString("createdAt"),
                            updatedAt = dataJson.getString("updatedAt")
                        )
                    } else null
                    
                    Result.success(BaseResponse(status, message, contact))
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
     * Xóa liên hệ khẩn cấp
     */
    suspend fun delete(accessToken: String, id: Long): Result<BaseResponse<Unit>> {
        return withContext(Dispatchers.IO) {
            try {
                val httpRequest = Request.Builder()
                    .url("$baseUrl/$id")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .delete()
                    .build()

                val response = client.newCall(httpRequest).execute()
                val responseBody = response.body?.string()

                if (response.isSuccessful && responseBody != null) {
                    val jsonResponse = JSONObject(responseBody)
                    val status = jsonResponse.getString("status")
                    val message = jsonResponse.getString("message")
                    
                    Result.success(BaseResponse(status, message, Unit))
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
