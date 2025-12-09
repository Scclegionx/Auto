package com.auto_fe.auto_fe.ui.service

import com.auto_fe.auto_fe.network.ApiConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class RelationshipService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val baseUrl = "${ApiConfig.BASE_URL.replace("/api", "")}/api/relationships"

    data class RelationshipRequest(
        val id: Long,
        val elderUserId: Long,
        val elderUserName: String,
        val elderUserEmail: String?,
        val elderUserAvatar: String?,
        val elderUserPhone: String?,
        val elderBloodType: String?,
        val elderHeight: Double?,
        val elderWeight: Double?,
        val elderGender: String?,
        val supervisorUserId: Long,
        val supervisorUserName: String,
        val supervisorUserEmail: String?,
        val supervisorUserAvatar: String?,
        val supervisorOccupation: String?,
        val supervisorWorkplace: String?,
        val requesterId: Long,
        val requesterName: String,
        val status: String,
        val requestMessage: String?,
        val responseMessage: String?,
        val respondedAt: String?,
        val createdAt: String
    )

    data class SendRequestDTO(
        val targetUserId: Long,
        val message: String?
    )

    data class RespondRequestDTO(
        val message: String?
    )

    /**
     * Gửi yêu cầu kết nối
     */
    suspend fun sendRequest(
        accessToken: String,
        targetUserId: Long,
        message: String?
    ): Result<RelationshipRequest> = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject().apply {
                put("targetUserId", targetUserId)
                if (message != null) put("message", message)
            }

            val requestBody = json.toString()
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("$baseUrl/request")
                .addHeader("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""

                if (!response.isSuccessful) {
                    val errorMsg = try {
                        JSONObject(responseBody).optString("message", "Gửi yêu cầu thất bại")
                    } catch (e: Exception) {
                        "Gửi yêu cầu thất bại: ${response.code}"
                    }
                    return@withContext Result.failure(Exception(errorMsg))
                }

                val jsonResponse = JSONObject(responseBody)
                Result.success(parseRelationshipRequest(jsonResponse))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Chấp nhận yêu cầu
     */
    suspend fun acceptRequest(
        accessToken: String,
        requestId: Long,
        message: String?
    ): Result<RelationshipRequest> = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject().apply {
                if (message != null) put("message", message)
            }

            val requestBody = json.toString()
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("$baseUrl/$requestId/accept")
                .addHeader("Authorization", "Bearer $accessToken")
                .put(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""

                if (!response.isSuccessful) {
                    val errorMsg = try {
                        JSONObject(responseBody).optString("message", "Chấp nhận thất bại")
                    } catch (e: Exception) {
                        "Chấp nhận thất bại: ${response.code}"
                    }
                    return@withContext Result.failure(Exception(errorMsg))
                }

                val jsonResponse = JSONObject(responseBody)
                Result.success(parseRelationshipRequest(jsonResponse))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Từ chối yêu cầu
     */
    suspend fun rejectRequest(
        accessToken: String,
        requestId: Long,
        message: String?
    ): Result<RelationshipRequest> = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject().apply {
                if (message != null) put("message", message)
            }

            val requestBody = json.toString()
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("$baseUrl/$requestId/reject")
                .addHeader("Authorization", "Bearer $accessToken")
                .put(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""

                if (!response.isSuccessful) {
                    val errorMsg = try {
                        JSONObject(responseBody).optString("message", "Từ chối thất bại")
                    } catch (e: Exception) {
                        "Từ chối thất bại: ${response.code}"
                    }
                    return@withContext Result.failure(Exception(errorMsg))
                }

                val jsonResponse = JSONObject(responseBody)
                Result.success(parseRelationshipRequest(jsonResponse))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Lấy danh sách yêu cầu pending đã nhận
     */
    suspend fun getPendingReceivedRequests(accessToken: String): Result<List<RelationshipRequest>> =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/pending/received")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string() ?: ""

                    if (!response.isSuccessful) {
                        return@withContext Result.failure(Exception("Lấy danh sách thất bại: ${response.code}"))
                    }

                    val jsonArray = JSONArray(responseBody)
                    val requests = mutableListOf<RelationshipRequest>()
                    for (i in 0 until jsonArray.length()) {
                        requests.add(parseRelationshipRequest(jsonArray.getJSONObject(i)))
                    }
                    Result.success(requests)
                }
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    /**
     * Lấy danh sách Elder đã kết nối (cho Supervisor)
     */
    suspend fun getConnectedElders(accessToken: String): Result<List<RelationshipRequest>> =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/elders")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string() ?: ""

                    if (!response.isSuccessful) {
                        return@withContext Result.failure(Exception("Lấy danh sách Elder thất bại: ${response.code}"))
                    }

                    val jsonArray = JSONArray(responseBody)
                    val requests = mutableListOf<RelationshipRequest>()
                    for (i in 0 until jsonArray.length()) {
                        requests.add(parseRelationshipRequest(jsonArray.getJSONObject(i)))
                    }
                    Result.success(requests)
                }
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    /**
     * Lấy danh sách Supervisor đã kết nối (cho Elder)
     */
    suspend fun getConnectedSupervisors(accessToken: String): Result<List<RelationshipRequest>> =
        withContext(Dispatchers.IO) {
            try {
                val request = Request.Builder()
                    .url("$baseUrl/supervisors")
                    .addHeader("Authorization", "Bearer $accessToken")
                    .get()
                    .build()

                client.newCall(request).execute().use { response ->
                    val responseBody = response.body?.string() ?: ""

                    if (!response.isSuccessful) {
                        return@withContext Result.failure(Exception("Lấy danh sách Supervisor thất bại: ${response.code}"))
                    }

                    val jsonArray = JSONArray(responseBody)
                    val requests = mutableListOf<RelationshipRequest>()
                    for (i in 0 until jsonArray.length()) {
                        requests.add(parseRelationshipRequest(jsonArray.getJSONObject(i)))
                    }
                    Result.success(requests)
                }
            } catch (e: Exception) {
                Result.failure(e)
            }
        }

    private fun parseRelationshipRequest(json: JSONObject): RelationshipRequest {
        return RelationshipRequest(
            id = json.getLong("id"),
            elderUserId = json.getLong("elderUserId"),
            elderUserName = json.optString("elderUserName", ""),
            elderUserEmail = json.optString("elderUserEmail", null),
            elderUserAvatar = json.optString("elderUserAvatar", null),
            elderUserPhone = json.optString("elderUserPhone", null),
            elderBloodType = json.optString("elderBloodType", null),
            elderHeight = if (json.has("elderHeight")) json.getDouble("elderHeight") else null,
            elderWeight = if (json.has("elderWeight")) json.getDouble("elderWeight") else null,
            elderGender = json.optString("elderGender", null),
            supervisorUserId = json.getLong("supervisorUserId"),
            supervisorUserName = json.optString("supervisorUserName", ""),
            supervisorUserEmail = json.optString("supervisorUserEmail", null),
            supervisorUserAvatar = json.optString("supervisorUserAvatar", null),
            supervisorOccupation = json.optString("supervisorOccupation", null),
            supervisorWorkplace = json.optString("supervisorWorkplace", null),
            requesterId = json.getLong("requesterId"),
            requesterName = json.optString("requesterName", ""),
            status = json.getString("status"),
            requestMessage = json.optString("requestMessage", null),
            responseMessage = json.optString("responseMessage", null),
            respondedAt = json.optString("respondedAt", null),
            createdAt = json.getString("createdAt")
        )
    }
}
