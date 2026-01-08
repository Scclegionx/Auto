package com.auto_fe.auto_fe.automation.device

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.provider.MediaStore
import android.util.Log
import org.json.JSONObject

class CameraAutomation(private val context: Context) {

    companion object {
        private const val TAG = "CameraAutomation"
        const val REQUEST_IMAGE_CAPTURE = 1001
        const val REQUEST_VIDEO_CAPTURE = 1002
    }

    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     * CAMERA_TYPE: "image" hoặc "video"
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing camera with entities: $entities")

        // Parse dữ liệu
        val cameraType = entities.optString("CAMERA_TYPE", "")

        // Routing logic đơn giản: so sánh trực tiếp
        return when (cameraType) {
            "image" -> {
                capturePhoto()
            }
            "video" -> {
                captureVideo()
            }
            else -> {
                throw Exception("Dạ, con không hỗ trợ loại camera này ạ.")
            }
        }
    }

    // ========== PHOTO CAPTURE ==========

    /**
     * Chụp ảnh sử dụng camera mặc định
     */
    fun capturePhoto(): String {
        return try {
            Log.d(TAG, "Starting photo capture")
            
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            
            // Kiểm tra xem có app nào có thể xử lý intent này không
            if (takePictureIntent.resolveActivity(context.packageManager) != null) {
                // Nếu context là Activity, có thể start trực tiếp
                if (context is Activity) {
                    context.startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                    Log.d(TAG, "Photo capture intent started successfully")
                } else {
                    // Nếu không phải Activity, cần flag để start từ background
                    takePictureIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(takePictureIntent)
                    Log.d(TAG, "Photo capture intent started from background")
                }
                "Dạ, đã mở ứng dụng camera để chụp ảnh ạ."
            } else {
                Log.e(TAG, "No camera app available")
                throw Exception("Dạ, con không tìm thấy ứng dụng camera ạ.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing photo: ${e.message}", e)
            throw Exception("Dạ, con không thể mở camera ạ.")
        }
    }

    // ========== VIDEO CAPTURE ==========

    /**
     * Quay video sử dụng camera mặc định
     */
    fun captureVideo(): String {
        return try {
            Log.d(TAG, "Starting video capture")
            
            val takeVideoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            
            if (takeVideoIntent.resolveActivity(context.packageManager) != null) {
                if (context is Activity) {
                    context.startActivityForResult(takeVideoIntent, REQUEST_VIDEO_CAPTURE)
                    Log.d(TAG, "Video capture intent started successfully")
                } else {
                    takeVideoIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(takeVideoIntent)
                    Log.d(TAG, "Video capture intent started from background")
                }
                "Dạ, đã mở ứng dụng camera để quay video ạ."
            } else {
                Log.e(TAG, "No video camera app available")
                throw Exception("Dạ, con không tìm thấy ứng dụng camera quay video ạ.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing video: ${e.message}", e)
            throw Exception("Dạ, con không thể mở camera quay video ạ.")
        }
    }
}
