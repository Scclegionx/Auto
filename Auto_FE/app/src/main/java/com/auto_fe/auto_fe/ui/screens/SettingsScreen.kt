package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.clickable
import com.auto_fe.auto_fe.ui.theme.*
import androidx.compose.ui.platform.LocalContext
import com.auto_fe.auto_fe.utils.common.SettingsManager
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.utils.be.SessionManager
import com.auto_fe.auto_fe.service.be.UserSettingService
import kotlinx.coroutines.launch
import android.util.Log

/**
 * Data class để lưu trữ thông tin setting hiển thị
 */
data class DisplaySetting(
    val settingKey: String,
    val name: String,
    val description: String,
    val settingType: String,
    val currentValue: String,
    val defaultValue: String,
    val possibleValues: String?,
    val isBoolean: Boolean = false
)

/**
 * Màn hình cài đặt
 */
@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val settingsManager = remember { SettingsManager(context) }
    val sessionManager = remember { SessionManager(context) }
    val settingService = remember { UserSettingService() }
    val scope = rememberCoroutineScope()

    // State
    var isLoading by remember { mutableStateOf(true) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var displaySettings by remember { mutableStateOf<List<DisplaySetting>>(emptyList()) }
    var settingsByType by remember { mutableStateOf<Map<String, List<DisplaySetting>>>(emptyMap()) }

    // Kiểm tra login status
    val isLoggedIn = remember { sessionManager.isLoggedIn() }
    val userRole = remember { sessionManager.getUserRole() }
    val accessToken = remember { sessionManager.getAccessToken() }

    // Load settings khi màn hình được hiển thị
    LaunchedEffect(Unit) {
        isLoading = true
        errorMessage = null

        try {
            // Xác định setting types cần lấy
            val settingTypes = if (isLoggedIn) {
                when (userRole) {
                    "ELDER" -> listOf("AUTO", "COMMON", "MEDICATION")
                    "SUPERVISOR" -> listOf("COMMON", "MEDICATION", "PRESCRIPTION")
                    else -> listOf("AUTO", "COMMON")
                }
            } else {
                listOf("AUTO", "COMMON")
            }

            // Gọi API GetSettingsByType
            val settingsResult = settingService.getSettingsByType(settingTypes)
            
            if (settingsResult.isSuccess) {
                val settingsData = settingsResult.getOrNull()?.data ?: emptyList()
                
                if (isLoggedIn && accessToken != null) {
                    // Đã login: lấy user settings từ API
                    val userSettingsResult = settingService.getUserSettings(accessToken)
                    
                    if (userSettingsResult.isSuccess) {
                        val userSettingsData = userSettingsResult.getOrNull()?.data
                        val userSettingsMap = userSettingsData?.settings?.associateBy { it.settingKey } ?: emptyMap()
                        
                        // Tạo display settings với giá trị từ user settings
                        val displayList = settingsData.map { setting ->
                            val userSetting = userSettingsMap[setting.settingKey]
                            val currentValue = userSetting?.value ?: setting.defaultValue ?: ""
                            
                            DisplaySetting(
                                settingKey = setting.settingKey ?: "",
                                name = setting.name ?: "",
                                description = setting.description ?: "",
                                settingType = setting.settingType ?: "",
                                currentValue = currentValue,
                                defaultValue = setting.defaultValue ?: "",
                                possibleValues = setting.possibleValues,
                                isBoolean = currentValue.lowercase() in listOf("on", "off", "true", "false", "1", "0")
                            )
                        }
                        
                        // QUAN TRỌNG: Lưu settings từ API vào SharedPreferences để các automation có thể đọc
                        val settingsToSave = settingsData.associate { setting ->
                            val userSetting = userSettingsMap[setting.settingKey]
                            val value = userSetting?.value ?: setting.defaultValue ?: ""
                            (setting.settingKey ?: "") to value
                        }
                        settingsManager.saveSettings(settingsToSave)
                        
                        // Lưu defaultValue để reset khi logout
                        val defaultValues = settingsData.associate { setting ->
                            (setting.settingKey ?: "") to setting.defaultValue
                        }
                        settingsManager.saveDefaultValues(defaultValues)
                        
                        displaySettings = displayList
                        settingsByType = displayList.groupBy { it.settingType }
                    } else {
                        errorMessage = userSettingsResult.exceptionOrNull()?.message ?: "Không thể tải settings"
                    }
                } else {
                    // Chưa login: lấy từ SharedPreferences hoặc dùng default_value
                    val displayList = settingsData.map { setting ->
                        val savedValue = settingsManager.getSettingValue(setting.settingKey ?: "")
                        val currentValue = savedValue ?: setting.defaultValue ?: ""
                        
                        // Lưu default_value vào SharedPreferences nếu chưa có
                        if (savedValue == null && setting.defaultValue != null) {
                            settingsManager.setSettingValue(setting.settingKey ?: "", setting.defaultValue)
                        }
                        
                        // Lưu defaultValue để reset khi logout
                        if (setting.defaultValue != null) {
                            settingsManager.saveDefaultValues(mapOf(Pair(setting.settingKey ?: "", setting.defaultValue)))
                        }
                        
                        DisplaySetting(
                            settingKey = setting.settingKey ?: "",
                            name = setting.name ?: "",
                            description = setting.description ?: "",
                            settingType = setting.settingType ?: "",
                            currentValue = currentValue,
                            defaultValue = setting.defaultValue ?: "",
                            possibleValues = setting.possibleValues,
                            isBoolean = currentValue.lowercase() in listOf("on", "off", "true", "false", "1", "0")
                        )
                    }
                    
                    displaySettings = displayList
                    settingsByType = displayList.groupBy { it.settingType }
                }
            } else {
                errorMessage = settingsResult.exceptionOrNull()?.message ?: "Không thể tải danh sách settings"
            }
        } catch (e: Exception) {
            Log.e("SettingsScreen", "Error loading settings", e)
            errorMessage = "Lỗi: ${e.message}"
        } finally {
            isLoading = false
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        AIBackgroundDeep,
                        AIBackgroundSoft
                    )
                )
            )
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            // Header
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.8f)
                ),
                shape = RoundedCornerShape(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "⚙️ Cài đặt",
                        style = MaterialTheme.typography.headlineMedium.copy(fontSize = 34.sp),
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = "Tùy chỉnh ứng dụng theo nhu cầu",
                        style = MaterialTheme.typography.bodyMedium.copy(fontSize = 22.sp),
                        color = DarkOnSurface.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Loading state
            if (isLoading) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = DarkPrimary)
                }
            }
            // Error state
            else if (errorMessage != null) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "⚠️",
                            style = MaterialTheme.typography.headlineLarge,
                            fontSize = 48.sp
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = errorMessage ?: "Có lỗi xảy ra",
                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 20.sp),
                            color = DarkOnSurface,
                            textAlign = TextAlign.Center
                        )
                    }
                }
            }
            // Settings List
            else {
                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Hiển thị settings theo từng type
                    settingsByType.forEach { (type, settings) ->
                        item {
                            SettingsSection(title = when (type) {
                                "AUTO" -> "Tự động hóa"
                                "COMMON" -> "Chung"
                                "MEDICATION" -> "Thuốc"
                                "PRESCRIPTION" -> "Đơn thuốc"
                                else -> "Cài đặt"
                            })
                        }
                        
                        settings.forEach { setting ->
                            item {
                                if (setting.isBoolean) {
                                    // Hiển thị Switch cho boolean settings
                                    val isChecked = setting.currentValue.lowercase() in listOf("on", "true", "1", "yes")
                                    SettingsItem(
                                        title = setting.name,
                                        subtitle = setting.description,
                                        isChecked = isChecked,
                                        onCheckedChange = { newValue ->
                                            val newValueStr = if (newValue) "on" else "off"
                                            scope.launch {
                                                if (isLoggedIn && accessToken != null) {
                                                    // Gọi API update
                                                    settingService.updateUserSetting(
                                                        accessToken,
                                                        setting.settingKey,
                                                        newValueStr
                                                    )
                                                } else {
                                                    // Lưu vào SharedPreferences
                                                    settingsManager.setSettingValue(setting.settingKey, newValueStr)
                                                }
                                                
                                                // Cập nhật UI
                                                displaySettings = displaySettings.map {
                                                    if (it.settingKey == setting.settingKey) {
                                                        it.copy(currentValue = newValueStr)
                                                    } else {
                                                        it
                                                    }
                                                }
                                                settingsByType = displaySettings.groupBy { it.settingType }
                                            }
                                        }
                                    )
                                } else if (setting.possibleValues != null) {
                                    // Hiển thị Dropdown cho settings có possibleValues
                                    val options = setting.possibleValues.split(",").map { it.trim() }
                                    SettingsDropdown(
                                        title = setting.name,
                                        subtitle = setting.description,
                                        selectedValue = setting.currentValue,
                                        options = options,
                                        onValueChange = { newValue ->
                                            scope.launch {
                                                if (isLoggedIn && accessToken != null) {
                                                    // Gọi API update
                                                    settingService.updateUserSetting(
                                                        accessToken,
                                                        setting.settingKey,
                                                        newValue
                                                    )
                                                } else {
                                                    // Lưu vào SharedPreferences
                                                    settingsManager.setSettingValue(setting.settingKey, newValue)
                                                }
                                                
                                                // Cập nhật UI
                                                displaySettings = displaySettings.map {
                                                    if (it.settingKey == setting.settingKey) {
                                                        it.copy(currentValue = newValue)
                                                    } else {
                                                        it
                                                    }
                                                }
                                                settingsByType = displaySettings.groupBy { it.settingType }
                                            }
                                        }
                                    )
                                } else {
                                    // Hiển thị Info cho settings khác
                                    SettingsInfoItem(
                                        title = setting.name,
                                        value = setting.currentValue
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SettingsSection(title: String) {
    Text(
        text = title,
        style = MaterialTheme.typography.titleSmall.copy(fontSize = 24.sp),
        color = DarkOnSurface.copy(alpha = 0.7f),
        fontWeight = FontWeight.Medium,
        modifier = Modifier.padding(horizontal = 4.dp, vertical = 12.dp)
    )
}

@Composable
fun SettingsItem(
    title: String,
    subtitle: String,
    isChecked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.7f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMedium.copy(fontSize = 26.sp, lineHeight = 32.sp),
                    color = DarkOnSurface,
                    fontWeight = FontWeight.Medium
                )
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 20.sp, lineHeight = 26.sp),
                    color = DarkOnSurface.copy(alpha = 0.7f)
                )
            }
            
            Switch(
                checked = isChecked,
                onCheckedChange = onCheckedChange,
                colors = SwitchDefaults.colors(
                    checkedThumbColor = Color.White,
                    checkedTrackColor = DarkPrimary,
                    uncheckedThumbColor = Color.Gray,
                    uncheckedTrackColor = Color.Gray.copy(alpha = 0.3f)
                )
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsDropdown(
    title: String,
    subtitle: String,
    selectedValue: String,
    options: List<String>,
    onValueChange: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.7f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium.copy(fontSize = 26.sp, lineHeight = 32.sp),
                color = DarkOnSurface,
                fontWeight = FontWeight.Medium
            )
            
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall.copy(fontSize = 20.sp, lineHeight = 26.sp),
                color = DarkOnSurface.copy(alpha = 0.7f)
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = !expanded }
            ) {
                OutlinedTextField(
                    value = selectedValue,
                    onValueChange = {},
                    readOnly = true,
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor(),
                    textStyle = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = DarkPrimary,
                        unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f)
                    )
                )
                
                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    options.forEach { option ->
                        DropdownMenuItem(
                            text = { Text(option, style = MaterialTheme.typography.bodyLarge.copy(fontSize = 22.sp)) },
                            onClick = {
                                onValueChange(option)
                                expanded = false
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SettingsInfoItem(
    title: String,
    value: String
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.7f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium.copy(fontSize = 26.sp, lineHeight = 32.sp),
                color = DarkOnSurface,
                fontWeight = FontWeight.Medium
            )
            
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 22.sp, lineHeight = 28.sp),
                color = DarkOnSurface.copy(alpha = 0.8f)
            )
        }
    }
}

