package com.auto_fe.auto_fe.automation.device

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.MediaStore
import android.util.Log
import android.app.Activity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts

class CameraAutomation(private val context: Context) {

    interface CameraCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "CameraAutomation"
        const val REQUEST_IMAGE_CAPTURE = 1001
        const val REQUEST_VIDEO_CAPTURE = 1002
    }

    // ========== PHOTO CAPTURE ==========

    /**
     * Chụp ảnh sử dụng camera mặc định
     */
    fun capturePhoto(callback: CameraCallback) {
        try {
            Log.d(TAG, "Starting photo capture")
            
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            
            // Kiểm tra xem có app nào có thể xử lý intent này không
            if (takePictureIntent.resolveActivity(context.packageManager) != null) {
                // Nếu context là Activity, có thể start trực tiếp
                if (context is Activity) {
                    context.startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                    Log.d(TAG, "Photo capture intent started successfully")
                    callback.onSuccess()
                } else {
                    // Nếu không phải Activity, cần flag để start từ background
                    takePictureIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(takePictureIntent)
                    Log.d(TAG, "Photo capture intent started from background")
                    callback.onSuccess()
                }
            } else {
                Log.e(TAG, "No camera app available")
                callback.onError("Không tìm thấy ứng dụng camera")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing photo: ${e.message}", e)
            callback.onError("Lỗi chụp ảnh: ${e.message}")
        }
    }

    /**
     * Chụp ảnh với Intent ACTION_STILL_IMAGE_CAMERA - Mặc định chọn com.transsion.camera
     */
    fun capturePhotoWithStillImageIntent(callback: CameraCallback) {
        try {
            Log.d(TAG, "Starting photo capture with STILL_IMAGE_CAMERA intent - targeting com.transsion.camera")
            
            val stillImageIntent = Intent(MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA)
            
            // Chỉ định package cụ thể để tránh chooser dialog
            stillImageIntent.setPackage("com.transsion.camera")
            
            // Kiểm tra xem package có tồn tại và có thể xử lý intent không
            if (stillImageIntent.resolveActivity(context.packageManager) != null) {
                if (context is Activity) {
                    context.startActivityForResult(stillImageIntent, REQUEST_IMAGE_CAPTURE)
                    Log.d(TAG, "Transsion camera app started successfully")
                    callback.onSuccess()
                } else {
                    stillImageIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(stillImageIntent)
                    Log.d(TAG, "Transsion camera app started from background")
                    callback.onSuccess()
                }
            } else {
                // Fallback: thử với ACTION_IMAGE_CAPTURE nếu STILL_IMAGE_CAMERA không khả dụng
                Log.w(TAG, "com.transsion.camera not available for STILL_IMAGE_CAMERA, trying ACTION_IMAGE_CAPTURE")
                val fallbackIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                fallbackIntent.setPackage("com.transsion.camera")
                
                if (fallbackIntent.resolveActivity(context.packageManager) != null) {
                    if (context is Activity) {
                        context.startActivityForResult(fallbackIntent, REQUEST_IMAGE_CAPTURE)
                        Log.d(TAG, "Transsion camera app started with fallback intent")
                        callback.onSuccess()
                    } else {
                        fallbackIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        context.startActivity(fallbackIntent)
                        Log.d(TAG, "Transsion camera app started with fallback intent from background")
                        callback.onSuccess()
                    }
                } else {
                    Log.e(TAG, "com.transsion.camera not available for any camera intent")
                    callback.onError("Không tìm thấy ứng dụng camera Transsion")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing photo with still image intent: ${e.message}", e)
            callback.onError("Lỗi chụp ảnh: ${e.message}")
        }
    }

    // ========== VIDEO CAPTURE ==========

    /**
     * Quay video sử dụng camera mặc định
     */
    fun captureVideo(callback: CameraCallback) {
        try {
            Log.d(TAG, "Starting video capture")
            
            val takeVideoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            
            if (takeVideoIntent.resolveActivity(context.packageManager) != null) {
                if (context is Activity) {
                    context.startActivityForResult(takeVideoIntent, REQUEST_VIDEO_CAPTURE)
                    Log.d(TAG, "Video capture intent started successfully")
                    callback.onSuccess()
                } else {
                    takeVideoIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(takeVideoIntent)
                    Log.d(TAG, "Video capture intent started from background")
                    callback.onSuccess()
                }
            } else {
                Log.e(TAG, "No video camera app available")
                callback.onError("Không tìm thấy ứng dụng camera quay video")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing video: ${e.message}", e)
            callback.onError("Lỗi quay video: ${e.message}")
        }
    }

    /**
     * Quay video với Intent ACTION_VIDEO_CAMERA - Mặc định chọn com.transsion.camera
     */
    fun captureVideoWithVideoCameraIntent(callback: CameraCallback) {
        try {
            Log.d(TAG, "Starting video capture with VIDEO_CAMERA intent - targeting com.transsion.camera")
            
            val videoCameraIntent = Intent(MediaStore.INTENT_ACTION_VIDEO_CAMERA)
            
            // Chỉ định package cụ thể để tránh chooser dialog
            videoCameraIntent.setPackage("com.transsion.camera")
            
            // Kiểm tra xem package có tồn tại và có thể xử lý intent không
            if (videoCameraIntent.resolveActivity(context.packageManager) != null) {
                if (context is Activity) {
                    context.startActivityForResult(videoCameraIntent, REQUEST_VIDEO_CAPTURE)
                    Log.d(TAG, "Transsion video camera app started successfully")
                    callback.onSuccess()
                } else {
                    videoCameraIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(videoCameraIntent)
                    Log.d(TAG, "Transsion video camera app started from background")
                    callback.onSuccess()
                }
            } else {
                // Fallback: thử với ACTION_VIDEO_CAPTURE nếu INTENT_ACTION_VIDEO_CAMERA không khả dụng
                Log.w(TAG, "com.transsion.camera not available for INTENT_ACTION_VIDEO_CAMERA, trying ACTION_VIDEO_CAPTURE")
                val fallbackIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
                fallbackIntent.setPackage("com.transsion.camera")
                
                if (fallbackIntent.resolveActivity(context.packageManager) != null) {
                    if (context is Activity) {
                        context.startActivityForResult(fallbackIntent, REQUEST_VIDEO_CAPTURE)
                        Log.d(TAG, "Transsion video camera app started with fallback intent")
                        callback.onSuccess()
                    } else {
                        fallbackIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        context.startActivity(fallbackIntent)
                        Log.d(TAG, "Transsion video camera app started with fallback intent from background")
                        callback.onSuccess()
                    }
                } else {
                    Log.e(TAG, "com.transsion.camera not available for any video camera intent")
                    callback.onError("Không tìm thấy ứng dụng camera quay video Transsion")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error capturing video with video camera intent: ${e.message}", e)
            callback.onError("Lỗi quay video: ${e.message}")
        }
    }

    // ========== CAMERA APPS DETECTION ==========

    /**
     * Kiểm tra xem có app camera nào khả dụng không
     */
    fun isCameraAvailable(): Boolean {
        return try {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            cameraIntent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking camera availability: ${e.message}", e)
            false
        }
    }

    /**
     * Kiểm tra xem có app video camera nào khả dụng không
     */
    fun isVideoCameraAvailable(): Boolean {
        return try {
            val videoIntent = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            videoIntent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking video camera availability: ${e.message}", e)
            false
        }
    }

    /**
     * Kiểm tra xem có app still image camera nào khả dụng không
     */
    fun isStillImageCameraAvailable(): Boolean {
        return try {
            val stillImageIntent = Intent(MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA)
            stillImageIntent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking still image camera availability: ${e.message}", e)
            false
        }
    }

    /**
     * Kiểm tra xem có app video camera chuyên dụng nào khả dụng không
     */
    fun isVideoCameraIntentAvailable(): Boolean {
        return try {
            val videoCameraIntent = Intent(MediaStore.INTENT_ACTION_VIDEO_CAMERA)
            videoCameraIntent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking video camera intent availability: ${e.message}", e)
            false
        }
    }

    // ========== CAMERA PERMISSIONS ==========

    /**
     * Kiểm tra xem có quyền camera không
     */
    fun hasCameraPermission(): Boolean {
        return try {
            context.checkSelfPermission(android.Manifest.permission.CAMERA) == 
                PackageManager.PERMISSION_GRANTED
        } catch (e: Exception) {
            Log.e(TAG, "Error checking camera permission: ${e.message}", e)
            false
        }
    }

    /**
     * Kiểm tra xem có quyền ghi storage không (để lưu ảnh/video)
     */
    fun hasStoragePermission(): Boolean {
        return try {
            val writePermission = context.checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
            val readPermission = context.checkSelfPermission(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            
            // Từ Android 10+, WRITE_EXTERNAL_STORAGE không cần thiết cho MediaStore
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                readPermission == PackageManager.PERMISSION_GRANTED
            } else {
                writePermission == PackageManager.PERMISSION_GRANTED
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error checking storage permission: ${e.message}", e)
            false
        }
    }

    // ========== UTILITY METHODS ==========

    /**
     * Lấy danh sách các app camera có sẵn
     */
    fun getAvailableCameraApps(): List<String> {
        val cameraApps = mutableListOf<String>()
        
        try {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            val packageManager = context.packageManager
            val activities = packageManager.queryIntentActivities(cameraIntent, 0)
            
            for (activity in activities) {
                val appName = activity.loadLabel(packageManager).toString()
                cameraApps.add(appName)
                Log.d(TAG, "Found camera app: $appName")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error getting camera apps: ${e.message}", e)
        }
        
        return cameraApps
    }

    /**
     * Mở camera app mặc định (không chỉ định loại cụ thể)
     */
    fun openDefaultCameraApp(callback: CameraCallback) {
        try {
            Log.d(TAG, "Opening default camera app")
            
            // Thử mở camera app mặc định
            val cameraIntent = Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
            
            if (cameraIntent.resolveActivity(context.packageManager) != null) {
                if (context is Activity) {
                    context.startActivity(cameraIntent)
                } else {
                    cameraIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(cameraIntent)
                }
                Log.d(TAG, "Default camera app opened successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No default camera app available")
                callback.onError("Không tìm thấy ứng dụng camera mặc định")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error opening default camera app: ${e.message}", e)
            callback.onError("Lỗi mở camera: ${e.message}")
        }
    }

    /**
     * Kiểm tra tất cả quyền cần thiết cho camera
     */
    fun checkAllPermissions(): Map<String, Boolean> {
        return mapOf(
            "camera" to hasCameraPermission(),
            "storage" to hasStoragePermission()
        )
    }

    /**
     * Lấy thông tin chi tiết về camera capabilities
     */
    fun getCameraInfo(): Map<String, Any> {
        return mapOf(
            "camera_available" to isCameraAvailable(),
            "video_camera_available" to isVideoCameraAvailable(),
            "still_image_camera_available" to isStillImageCameraAvailable(),
            "video_camera_intent_available" to isVideoCameraIntentAvailable(),
            "camera_permission" to hasCameraPermission(),
            "storage_permission" to hasStoragePermission(),
            "available_apps" to getAvailableCameraApps()
        )
    }
}
