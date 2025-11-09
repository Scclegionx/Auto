package com.auto_fe.auto_fe.ui.service

import android.util.Log
import com.auto_fe.auto_fe.network.ApiConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class OcrService {
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private val baseUrl = ApiConfig.BASE_URL + "/ocr"

    suspend fun extractPrescriptionFromImage(
        imageFile: File,
        accessToken: String
    ): Result<OcrResult> = withContext(Dispatchers.IO) {
        try {
            Log.d("OcrService", "Extracting prescription: ${imageFile.name}")

            // Detect MIME type
            val mimeType = when (imageFile.extension.lowercase()) {
                "jpg", "jpeg" -> "image/jpeg"
                "png" -> "image/png"
                "gif" -> "image/gif"
                "webp" -> "image/webp"
                else -> "image/jpeg"
            }

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "image",
                    imageFile.name,
                    imageFile.asRequestBody(mimeType.toMediaType())
                )
                .build()

            val request = Request.Builder()
                .url("$baseUrl/extract-prescription")
                .addHeader("Authorization", "Bearer $accessToken")
                .post(requestBody)
                .build()

            val response = client.newCall(request).execute()
            val responseBody = response.body?.string()

            Log.d("OcrService", "Response: ${response.code}")

            if (response.isSuccessful && responseBody != null) {
                val json = JSONObject(responseBody)
                val data = json.getJSONObject("data")

                val medications = mutableListOf<OcrMedication>()
                val medsArray = data.getJSONArray("medicationReminders")
                
                for (i in 0 until medsArray.length()) {
                    val medJson = medsArray.getJSONObject(i)
                    val timesArray = medJson.getJSONArray("reminderTimes")
                    val times = mutableListOf<String>()
                    for (j in 0 until timesArray.length()) {
                        times.add(timesArray.getString(j))
                    }

                    medications.add(
                        OcrMedication(
                            name = medJson.getString("name"),
                            description = medJson.optString("description", ""),
                            type = medJson.getString("type"),
                            reminderTimes = times,
                            daysOfWeek = medJson.getString("daysOfWeek")
                        )
                    )
                }

                return@withContext Result.success(
                    OcrResult(
                        name = data.getString("name"),
                        description = data.getString("description"),
                        imageUrl = data.getString("imageUrl"),
                        medications = medications
                    )
                )
            } else {
                val errorMsg = try {
                    JSONObject(responseBody ?: "{}").optString("message", "OCR failed")
                } catch (e: Exception) {
                    "OCR failed"
                }
                return@withContext Result.failure(Exception(errorMsg))
            }
        } catch (e: Exception) {
            Log.e("OcrService", "Error", e)
            return@withContext Result.failure(Exception("Lỗi: ${e.message}"))
        }
    }
}

data class OcrResult(
    val name: String,
    val description: String,
    val imageUrl: String,
    val medications: List<OcrMedication>
)

data class OcrMedication(
    val name: String,
    val description: String,
    val type: String,
    val reminderTimes: List<String>,
    val daysOfWeek: String
)
