package com.auto_fe.auto_fe

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.auto_fe.auto_fe.ui.FloatingWindow
import com.auto_fe.auto_fe.ui.theme.Auto_FETheme
import com.auto_fe.auto_fe.utils.PermissionManager
import com.auto_fe.auto_fe.automation.msg.WAAutomation
import com.auto_fe.auto_fe.automation.alarm.AlarmAutomation
import com.auto_fe.auto_fe.automation.calendar.CalendarAutomation
import com.auto_fe.auto_fe.automation.third_apps.YouTubeAutomation
import com.auto_fe.auto_fe.automation.third_apps.ChromeAutomation
import android.util.Log

class MainActivity : ComponentActivity() {
    private lateinit var permissionManager: PermissionManager
    private lateinit var floatingWindow: FloatingWindow

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            checkOverlayPermission()
        } else {
            Toast.makeText(this, "Cần cấp quyền để sử dụng ứng dụng", Toast.LENGTH_LONG).show()
        }
    }

    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (Settings.canDrawOverlays(this)) {
                startFloatingWindow()
            } else {
                Toast.makeText(this, "Cần cấp quyền hiển thị trên các ứng dụng khác", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        permissionManager = PermissionManager(this)
        floatingWindow = FloatingWindow(this)

        setContent {
            Auto_FETheme {
                MainScreen()
            }
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        if (!permissionManager.checkAllPermissions()) {
            requestPermissionLauncher.launch(permissionManager.getMissingPermissions().toTypedArray())
        } else {
            checkOverlayPermission()
        }
    }

    private fun checkOverlayPermission() {
        if (!permissionManager.checkOverlayPermission()) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            overlayPermissionLauncher.launch(intent)
        } else {
            startFloatingWindow()
        }
    }

    private fun startFloatingWindow() {
        floatingWindow.showFloatingWindow()
        Toast.makeText(this, "Auto FE đã sẵn sàng!", Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        floatingWindow.hideFloatingWindow()
        // Giải phóng resources để tránh memory leak
        floatingWindow.release()
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Auto FE",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "Ứng dụng tự động hóa thao tác điện thoại",
                style = MaterialTheme.typography.bodyLarge,
                modifier = Modifier.padding(horizontal = 16.dp)
            )

            Spacer(modifier = Modifier.height(32.dp))

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Hướng dẫn sử dụng:",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "1. Cấp quyền cần thiết cho ứng dụng\n" +
                                "2. Cửa sổ nổi sẽ xuất hiện\n" +
                                "3. Nhấn vào cửa sổ nổi để mở menu\n" +
                                "4. Chọn 'Ghi âm lệnh' để bắt đầu",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Button test WhatsApp
            Button(
                onClick = {
                    testWhatsAppFunction(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text("Test WhatsApp")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button test Alarm
            Button(
                onClick = {
                    testAlarmFunction(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text("Test Alarm")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button test Calendar
            Button(
                onClick = {
                    testCalendarFunction(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text("Test Calendar")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button test YouTube
            Button(
                onClick = {
                    testYouTubeFunction(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text("Test YouTube")
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Button test Chrome
            Button(
                onClick = {
                    testChromeFunction(context)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
            ) {
                Text("Test Chrome")
            }
        }
    }
}

private fun testWhatsAppFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting WhatsApp test...")
    
    // Test trực tiếp với WAAutomation
    val waAutomation = WAAutomation(context)
    waAutomation.sendWA("mẹ", "con sắp về", object : WAAutomation.WACallback {
        override fun onSuccess() {
            Log.d("MainActivity", "WhatsApp test successful!")
            android.widget.Toast.makeText(context, "Test WhatsApp thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "WhatsApp test error: $error")
            android.widget.Toast.makeText(context, "Test WhatsApp lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testAlarmFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Alarm test...")

    // Test trực tiếp với AlarmAutomation
    val alarmAutomation = AlarmAutomation(context)
    
    // Tạo báo thức mặc định (9h sáng thứ 2 hàng tuần)
    alarmAutomation.createDefaultAlarm(object : AlarmAutomation.AlarmCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Alarm test successful!")
            android.widget.Toast.makeText(context, "Đã tạo báo thức thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Alarm test error: $error")
            android.widget.Toast.makeText(context, "Test Alarm lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testCalendarFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Calendar test...")

    // Kiểm tra quyền calendar
    if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_CALENDAR) 
        != PackageManager.PERMISSION_GRANTED) {
        
        Log.d("MainActivity", "Calendar permission not granted")
        android.widget.Toast.makeText(context, "Cần quyền truy cập lịch để tạo sự kiện", android.widget.Toast.LENGTH_LONG).show()
        return
    }

    // Test trực tiếp với CalendarAutomation
    val calendarAutomation = CalendarAutomation(context)
    
    // Tạo sự kiện mặc định (Họp thứ 4 tới lúc 10h sáng)
    calendarAutomation.createDefaultEvent(object : CalendarAutomation.CalendarCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Calendar test successful!")
            android.widget.Toast.makeText(context, "Đã tạo sự kiện thành công!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Calendar test error: $error")
            android.widget.Toast.makeText(context, "Test Calendar lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testYouTubeFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting YouTube test...")

    // Test trực tiếp với YouTubeAutomation
    val youtubeAutomation = YouTubeAutomation(context)
    
    // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
    youtubeAutomation.searchDefault(object : YouTubeAutomation.YouTubeCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "YouTube test successful!")
            android.widget.Toast.makeText(context, "Đã mở YouTube tìm kiếm!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "YouTube test error: $error")
            android.widget.Toast.makeText(context, "Test YouTube lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}

private fun testChromeFunction(context: android.content.Context) {
    Log.d("MainActivity", "Starting Chrome test...")

    // Test trực tiếp với ChromeAutomation
    val chromeAutomation = ChromeAutomation(context)
    
    // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
    chromeAutomation.searchDefault(object : ChromeAutomation.ChromeCallback {
        override fun onSuccess() {
            Log.d("MainActivity", "Chrome test successful!")
            android.widget.Toast.makeText(context, "Đã mở Chrome tìm kiếm!", android.widget.Toast.LENGTH_LONG).show()
        }
        override fun onError(error: String) {
            Log.e("MainActivity", "Chrome test error: $error")
            android.widget.Toast.makeText(context, "Test Chrome lỗi: $error", android.widget.Toast.LENGTH_LONG).show()
        }
    })
}