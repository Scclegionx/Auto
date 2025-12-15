package com.auto_fe.auto_fe.ui.service

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.LocalDateTime

data class MedicalDocumentData(
    val id: Long,
    val name: String,
    val description: String,
    val elderUserId: Long,
    val elderUserName: String,
    val createdAt: String,
    val updatedAt: String,
    val files: List<MedicalDocumentFileData>,
    val fileCount: Int
)

data class MedicalDocumentFileData(
    val id: Long,
    val fileName: String,
    val fileType: String,
    val fileUrl: String,
    val fileSize: Int,
    val note: String?,
    val medicalDocumentId: Long,
    val createdAt: String
)

class MedicalDocumentService {
    private val client = OkHttpClient()
    private val baseUrl = com.auto_fe.auto_fe.network.ApiConfig.BASE_URL + "/medical-documents"

    /**
     * Lấy danh sách tài liệu y tế
     */
    suspend fun getDocuments(
        accessToken: String,
        elderUserId: Long? = null
    ): Result<List<MedicalDocumentData>> = withContext(Dispatchers.IO) {
        try {
            val url = if (elderUserId != null) {
                "$baseUrl?elderUserId=$elderUserId"
            } else {
                baseUrl
            }
            
            val request = Request.Builder()
                .url(url)
                .addHeader("Authorization", "Bearer $accessToken")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val dataArray = jsonResponse.getJSONArray("data")
                
                val documents = mutableListOf<MedicalDocumentData>()
                for (i in 0 until dataArray.length()) {
                    val item = dataArray.getJSONObject(i)
                    documents.add(parseDocument(item))
                }
                
                Result.success(documents)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Tạo tài liệu y tế mới
     */
    suspend fun createDocument(
        accessToken: String,
        name: String,
        description: String,
        elderUserId: Long? = null
    ): Result<MedicalDocumentData> = withContext(Dispatchers.IO) {
        try {
            val jsonBody = JSONObject().apply {
                put("name", name)
                put("description", description)
                if (elderUserId != null) {
                    put("elderUserId", elderUserId)
                }
            }
            
            val requestBody = jsonBody.toString()
                .toRequestBody("application/json".toMediaTypeOrNull())
            
            val request = Request.Builder()
                .url(baseUrl)
                .addHeader("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val data = jsonResponse.getJSONObject("data")
                
                Result.success(parseDocument(data))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Upload file vào tài liệu
     */
    suspend fun uploadFile(
        accessToken: String,
        documentId: Long,
        file: File,
        fileType: String,
        note: String? = null
    ): Result<MedicalDocumentFileData> = withContext(Dispatchers.IO) {
        try {
            // Detect correct MIME type
            val mimeType = when {
                fileType.contains("image") -> fileType
                file.name.endsWith(".pdf", ignoreCase = true) -> "application/pdf"
                file.name.endsWith(".jpg", ignoreCase = true) || 
                file.name.endsWith(".jpeg", ignoreCase = true) -> "image/jpeg"
                file.name.endsWith(".png", ignoreCase = true) -> "image/png"
                file.name.endsWith(".webp", ignoreCase = true) -> "image/webp"
                else -> fileType
            }
            
            android.util.Log.d("MedicalDocumentService", "Uploading file: ${file.name}, mimeType: $mimeType")
            
            val fileRequestBody = file.asRequestBody(mimeType.toMediaTypeOrNull())
            
            val multipartBuilder = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    file.name,
                    fileRequestBody
                )
            
            if (note != null) {
                multipartBuilder.addFormDataPart("note", note)
            }
            
            val requestBody = multipartBuilder.build()
            
            val request = Request.Builder()
                .url("$baseUrl/$documentId/files")
                .addHeader("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                android.util.Log.d("MedicalDocumentService", "Upload response code: ${response.code}")
                android.util.Log.d("MedicalDocumentService", "Upload response body: $responseBody")
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                try {
                    val jsonResponse = JSONObject(responseBody)
                    
                    // Kiểm tra status
                    val status = jsonResponse.optString("status", "")
                    if (status == "failed") {
                        val errorMsg = jsonResponse.optString("message", "Upload thất bại")
                        return@withContext Result.failure(Exception(errorMsg))
                    }
                    
                    if (!jsonResponse.has("data") || jsonResponse.isNull("data")) {
                        val msg = jsonResponse.optString("message", "Backend không trả về dữ liệu")
                        return@withContext Result.failure(Exception(msg))
                    }
                    
                    val data = jsonResponse.getJSONObject("data")
                    Result.success(parseDocumentFile(data))
                } catch (e: Exception) {
                    android.util.Log.e("MedicalDocumentService", "Parse error: ${e.message}")
                    return@withContext Result.failure(
                        Exception("Lỗi parse response: ${e.message}")
                    )
                }
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Cập nhật tài liệu
     */
    suspend fun updateDocument(
        accessToken: String,
        documentId: Long,
        name: String,
        description: String
    ): Result<MedicalDocumentData> = withContext(Dispatchers.IO) {
        try {
            val jsonBody = JSONObject().apply {
                put("name", name)
                put("description", description)
            }
            
            val requestBody = jsonBody.toString()
                .toRequestBody("application/json".toMediaTypeOrNull())
            
            val request = Request.Builder()
                .url("$baseUrl/$documentId")
                .addHeader("Authorization", "Bearer $accessToken")
                .put(requestBody)
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val data = jsonResponse.getJSONObject("data")
                
                Result.success(parseDocument(data))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Xóa tài liệu
     */
    suspend fun deleteDocument(
        accessToken: String,
        documentId: Long
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/$documentId")
                .addHeader("Authorization", "Bearer $accessToken")
                .delete()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val message = jsonResponse.getString("message")
                
                Result.success(message)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Xóa file khỏi tài liệu
     */
    suspend fun deleteFile(
        accessToken: String,
        documentId: Long,
        fileId: Long
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/$documentId/files/$fileId")
                .addHeader("Authorization", "Bearer $accessToken")
                .delete()
                .build()

            client.newCall(request).execute().use { response ->
                val responseBody = response.body?.string() ?: ""
                
                if (!response.isSuccessful) {
                    return@withContext Result.failure(
                        Exception(parseErrorMessage(responseBody))
                    )
                }

                val jsonResponse = JSONObject(responseBody)
                val message = jsonResponse.getString("message")
                
                Result.success(message)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    // Helper functions
    private fun parseDocument(json: JSONObject): MedicalDocumentData {
        val filesArray = json.optJSONArray("files") ?: JSONArray()
        val files = mutableListOf<MedicalDocumentFileData>()
        for (i in 0 until filesArray.length()) {
            files.add(parseDocumentFile(filesArray.getJSONObject(i)))
        }
        
        return MedicalDocumentData(
            id = json.getLong("id"),
            name = json.getString("name"),
            description = json.getString("description"),
            elderUserId = json.getLong("elderUserId"),
            elderUserName = json.getString("elderUserName"),
            createdAt = json.getString("createdAt"),
            updatedAt = json.getString("updatedAt"),
            files = files,
            fileCount = json.getInt("fileCount")
        )
    }
    
    private fun parseDocumentFile(json: JSONObject): MedicalDocumentFileData {
        return MedicalDocumentFileData(
            id = json.getLong("id"),
            fileName = json.getString("fileName"),
            fileType = json.getString("fileType"),
            fileUrl = json.getString("fileUrl"),
            fileSize = json.getInt("fileSize"),
            note = json.optString("note", null),
            medicalDocumentId = json.getLong("medicalDocumentId"),
            createdAt = json.getString("createdAt")
        )
    }

    private fun parseErrorMessage(responseBody: String): String {
        return try {
            val jsonResponse = JSONObject(responseBody)
            jsonResponse.optString("message", "Lỗi không xác định")
        } catch (e: Exception) {
            "Lỗi kết nối: ${e.message}"
        }
    }
}
