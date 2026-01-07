package com.auto_fe.auto_fe.utils.common

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class PermissionManager(private val context: Context) {
    
    companion object {
        const val PERMISSION_REQUEST_CODE = 1001
        const val OVERLAY_PERMISSION_REQUEST_CODE = 1002
    }
    
    private val requiredPermissions = arrayOf(
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.SEND_SMS,
        Manifest.permission.READ_SMS,
        Manifest.permission.CALL_PHONE,
        Manifest.permission.READ_PHONE_STATE,
        Manifest.permission.READ_CONTACTS,
        Manifest.permission.WRITE_CONTACTS,
        Manifest.permission.READ_CALL_LOG,
        Manifest.permission.WRITE_CALENDAR
    )
    
    fun checkAllPermissions(): Boolean {
        return requiredPermissions.all { permission ->
            ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    fun checkOverlayPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Settings.canDrawOverlays(context)
        } else {
            true
        }
    }
    
    fun requestPermissions(activity: Activity) {
        val permissionsToRequest = requiredPermissions.filter { permission ->
            ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                activity,
                permissionsToRequest.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        }
    }
    
    fun requestOverlayPermission(activity: Activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.canDrawOverlays(context)) {
                val intent = Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:${context.packageName}")
                )
                activity.startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
            }
        }
    }
    
    fun onPermissionResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ): Boolean {
        return when (requestCode) {
            PERMISSION_REQUEST_CODE -> {
                grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            }
            else -> false
        }
    }
    
    fun getMissingPermissions(): List<String> {
        return requiredPermissions.filter { permission ->
            ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED
        }
    }
    
    // ========== CAMERA PERMISSIONS ==========
    
    /**
     * Kiểm tra xem có quyền camera không
     */
    fun hasCameraPermission(): Boolean {
        return try {
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == 
                PackageManager.PERMISSION_GRANTED
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Kiểm tra xem có quyền ghi storage không (để lưu ảnh/video)
     */
    fun hasStoragePermission(): Boolean {
        return try {
            val writePermission = ContextCompat.checkSelfPermission(
                context, 
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            val readPermission = ContextCompat.checkSelfPermission(
                context, 
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
            
            // Từ Android 10+, WRITE_EXTERNAL_STORAGE không cần thiết cho MediaStore
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                readPermission == PackageManager.PERMISSION_GRANTED
            } else {
                writePermission == PackageManager.PERMISSION_GRANTED
            }
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Kiểm tra tất cả quyền cần thiết cho camera
     */
    fun checkAllCameraPermissions(): Map<String, Boolean> {
        return mapOf(
            "camera" to hasCameraPermission(),
            "storage" to hasStoragePermission()
        )
    }
}

