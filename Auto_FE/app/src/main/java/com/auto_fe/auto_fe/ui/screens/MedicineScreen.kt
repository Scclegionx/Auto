package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auto_fe.auto_fe.ui.theme.*

/**
 * Màn hình quản lý thuốc
 */
@Composable
fun MedicineScreen() {
    // Sample data - sẽ được thay thế bằng data từ API
    val medicines = remember {
        listOf(
            MedicineItem(
                id = "1",
                name = "Paracetamol 500mg",
                dosage = "2 viên/ngày",
                time = "8:00, 20:00",
                status = "Đang dùng"
            ),
            MedicineItem(
                id = "2", 
                name = "Vitamin D3",
                dosage = "1 viên/ngày",
                time = "7:00",
                status = "Đang dùng"
            ),
            MedicineItem(
                id = "3",
                name = "Omega 3",
                dosage = "1 viên/ngày", 
                time = "19:00",
                status = "Hết thuốc"
            )
        )
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
                        text = "💊 Quản lý thuốc",
                        style = MaterialTheme.typography.headlineMedium,
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    
                    Spacer(modifier = Modifier.height(8.dp))
                    
                    Text(
                        text = "Theo dõi lịch uống thuốc hàng ngày",
                        style = MaterialTheme.typography.bodyMedium,
                        color = DarkOnSurface.copy(alpha = 0.8f),
                        textAlign = TextAlign.Center
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Medicine List
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(medicines) { medicine ->
                    MedicineCard(medicine = medicine)
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Add Medicine Button
            Button(
                onClick = { /* TODO: Navigate to add medicine */ },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkPrimary
                ),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(
                    text = "➕ Thêm thuốc mới",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun MedicineCard(medicine: MedicineItem) {
    val statusColor = when (medicine.status) {
        "Đang dùng" -> VoiceListening
        "Hết thuốc" -> Color.Red.copy(alpha = 0.8f)
        "Tạm dừng" -> Color(0xFFFF9800).copy(alpha = 0.8f)
        else -> DarkOnSurface.copy(alpha = 0.6f)
    }

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
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = medicine.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = DarkOnSurface,
                    fontWeight = FontWeight.Bold
                )
                
                Card(
                    colors = CardDefaults.cardColors(containerColor = statusColor),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = medicine.status,
                        style = MaterialTheme.typography.labelSmall,
                        color = Color.White,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                        text = "Liều lượng:",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                    Text(
                        text = medicine.dosage,
                        style = MaterialTheme.typography.bodyMedium,
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Medium
                    )
                }
                
                Column {
                    Text(
                        text = "Thời gian:",
                        style = MaterialTheme.typography.bodySmall,
                        color = DarkOnSurface.copy(alpha = 0.7f)
                    )
                    Text(
                        text = medicine.time,
                        style = MaterialTheme.typography.bodyMedium,
                        color = DarkOnSurface,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

data class MedicineItem(
    val id: String,
    val name: String,
    val dosage: String,
    val time: String,
    val status: String
)
