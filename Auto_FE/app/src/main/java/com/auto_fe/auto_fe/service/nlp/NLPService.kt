package com.auto_fe.auto_fe.service.nlp

import android.content.Context
import android.util.Log
import com.auto_fe.auto_fe.config.nlp.NlpConfig
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
    
    private val serverUrl = NlpConfig.NLP_SERVER_URL

    suspend fun analyzeText(text: String): JSONObject = withContext(Dispatchers.IO) {
        val json = JSONObject().apply { put("text", text) }
        val request = Request.Builder()
            .url(serverUrl)
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        // OkHttp hỗ trợ execute() trong blocking thread, kết hợp với withContext là chuẩn nhất
        val response = client.newCall(request).execute()
        
        if (!response.isSuccessful) throw Exception("Server error: ${response.code}")
        
        val body = response.body?.string() ?: throw Exception("Empty response")
        return@withContext JSONObject(body)
    }
}

