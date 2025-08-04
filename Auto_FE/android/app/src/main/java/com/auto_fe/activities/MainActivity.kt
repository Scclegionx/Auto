package com.auto_fe.activities

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.auto_fe.services.FloatingWidgetService
import com.facebook.react.ReactActivity
import com.facebook.react.ReactActivityDelegate
import com.facebook.react.defaults.DefaultNewArchitectureEntryPoint.fabricEnabled
import com.facebook.react.defaults.DefaultReactActivityDelegate

class MainActivity : ReactActivity() {

  /**
   * Returns the name of the main component registered from JavaScript. This is used to schedule
   * rendering of the component.
   */
  override fun getMainComponentName(): String = "Auto_FE"

  /**
   * Returns the instance of the [ReactActivityDelegate]. We use [DefaultReactActivityDelegate]
   * which allows you to enable New Architecture with a single boolean flags [fabricEnabled]
   */
  override fun createReactActivityDelegate(): ReactActivityDelegate =
      DefaultReactActivityDelegate(this, mainComponentName, fabricEnabled)

  override fun onCreate(savedInstanceState: android.os.Bundle?) {
    super.onCreate(savedInstanceState)
    
    // Kiểm tra và yêu cầu quyền ghi âm và notification
    val permissions = mutableListOf<String>()
    
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
      permissions.add(Manifest.permission.RECORD_AUDIO)
    }
    
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && 
        ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
      permissions.add(Manifest.permission.POST_NOTIFICATIONS)
    }
    
    if (permissions.isNotEmpty()) {
      ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSIONS_REQUEST_CODE)
    } else {
      checkOverlayPermission()
    }
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    when (requestCode) {
      PERMISSIONS_REQUEST_CODE -> {
        var allGranted = true
        for (i in permissions.indices) {
          if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
            allGranted = false
            break
          }
        }
        
        if (allGranted) {
          checkOverlayPermission()
        } else {
          Toast.makeText(this, "Cần quyền ghi âm và notification để sử dụng đầy đủ tính năng", Toast.LENGTH_LONG).show()
        }
      }
    }
  }

  override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
    super.onActivityResult(requestCode, resultCode, data)
    if (requestCode == OVERLAY_PERMISSION_REQUEST_CODE) {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && Settings.canDrawOverlays(this)) {
        startFloatingWidget()
      } else {
        Toast.makeText(this, "Cần quyền hiển thị trên ứng dụng khác để sử dụng widget nổi", Toast.LENGTH_LONG).show()
      }
    }
  }

  private fun checkOverlayPermission() {
    // Kiểm tra và yêu cầu quyền overlay
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
      val intent = Intent(
        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
        Uri.parse("package:$packageName")
      )
      startActivityForResult(intent, OVERLAY_PERMISSION_REQUEST_CODE)
    } else {
      startFloatingWidget()
    }
  }

  private fun startFloatingWidget() {
    val intent = Intent(this, FloatingWidgetService::class.java)
    startService(intent)
  }

  companion object {
    private const val OVERLAY_PERMISSION_REQUEST_CODE = 1234
    private const val PERMISSIONS_REQUEST_CODE = 5678
  }
}
