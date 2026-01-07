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

/**
 * Màn hình cài đặt
 */
@Composable
fun SettingsScreen() {
    val context = LocalContext.current
    val settingsManager = remember { SettingsManager(context) }
    var isSupportSpeakEnabled by remember { mutableStateOf(settingsManager.isSupportSpeakEnabled()) }

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

            // Settings List
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    SettingsSection(title = "Cài đặt")
                }
                
                // Toggle Hỗ trợ nói
                item {
                    SettingsItem(
                        title = "Hỗ trợ nói",
                        subtitle = "Thêm bước xác nhận ở một số lệnh",
                        isChecked = isSupportSpeakEnabled,
                        onCheckedChange = {
                            isSupportSpeakEnabled = it
                            settingsManager.setSupportSpeakEnabled(it)
                        }
                    )
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

@Composable
fun SettingsActionItem(
    title: String,
    subtitle: String,
    onClick: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = DarkSurface.copy(alpha = 0.7f)
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() }
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
            
            Text(
                text = "→",
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 36.sp),
                color = DarkPrimary
            )
        }
    }
}
