package com.auto_fe.auto_fe.service

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.utils.Config
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

class NLPService(private val context: Context) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()
    
    private val serverUrl = Config.NLP_SERVER_URL
    
    data class NLPResponse(
        val command: String,
        val ent: String,
        val `val`: String,
        val rawJson: String
    )
    
    interface NLPServiceCallback {
        fun onSuccess(response: NLPResponse)
        fun onError(error: String)
    }
    
    suspend fun sendCommandToServer(command: String, callback: NLPServiceCallback) {
        withContext(Dispatchers.IO) {
            try {
                val json = JSONObject().apply {
                    put("text", command)
                }
                
                val requestBody = json.toString()
                    .toRequestBody("application/json".toMediaType())
                
                val request = Request.Builder()
                    .url(serverUrl)
                    .post(requestBody)
                    .addHeader("Content-Type", "application/json")
                    .build()
                
                val response = client.newCall(request).execute()
                
                Log.d("NLPService", "Response code: ${response.code}")
                Log.d("NLPService", "Response message: ${response.message}")
                Log.d("NLPService", "Response headers: ${response.headers}")
                
                if (response.isSuccessful) {
                    val responseBody = response.body?.string()
                    if (responseBody != null) {
                        val jsonResponse = JSONObject(responseBody)
                        val nlpResponse = NLPResponse(
                            command = jsonResponse.optString("command", ""),
                            ent = jsonResponse.optString("ent", "{}"),
                            `val` = jsonResponse.optString("val", "{}"),
                            rawJson = responseBody
                        )
                        
                        withContext(Dispatchers.Main) {
                            callback.onSuccess(nlpResponse)
                        }
                    } else {
                        withContext(Dispatchers.Main) {
                            callback.onError("Response body is null")
                        }
                    }
                } else {
                    val errorBody = response.body?.string() ?: "No error body"
                    Log.e("NLPService", "Server error body: $errorBody")
                    withContext(Dispatchers.Main) {
                        callback.onError("Server error: ${response.code} - ${response.message}. Body: $errorBody")
                    }
                }
                
            } catch (e: IOException) {
                Log.e("NLPService", "Network error", e)
                withContext(Dispatchers.Main) {
                    callback.onError("Lỗi kết nối: ${e.message}")
                }
            } catch (e: Exception) {
                Log.e("NLPService", "Unexpected error", e)
                withContext(Dispatchers.Main) {
                    callback.onError("Lỗi không xác định: ${e.message}")
                }
            }
        }
    }
    
    fun testConnection(callback: NLPServiceCallback) {
        CoroutineScope(Dispatchers.Main).launch {
            sendCommandToServer("test", callback)
        }
    }
}
