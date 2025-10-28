package com.auto_fe.auto_fe.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auto_fe.auto_fe.ui.service.ChangePasswordService
import com.auto_fe.auto_fe.ui.theme.*
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChangePasswordScreen(
    accessToken: String,
    onBackClick: () -> Unit = {},
    onSuccess: () -> Unit = {}
) {
    val changePasswordService = remember { ChangePasswordService() }
    val coroutineScope = rememberCoroutineScope()
    
    var currentPassword by remember { mutableStateOf("") }
    var newPassword by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    
    var currentPasswordVisible by remember { mutableStateOf(false) }
    var newPasswordVisible by remember { mutableStateOf(false) }
    var confirmPasswordVisible by remember { mutableStateOf(false) }
    
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var successMessage by remember { mutableStateOf<String?>(null) }

    // Validation errors
    var passwordError by remember { mutableStateOf<String?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Đổi mật khẩu") },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(Icons.Default.ArrowBack, "Quay lại")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = DarkSurface,
                    titleContentColor = DarkOnSurface,
                    navigationIconContentColor = DarkOnSurface
                )
            )
        },
        containerColor = DarkBackground
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(24.dp))
            
            // Icon
            Surface(
                modifier = Modifier.size(100.dp),
                shape = RoundedCornerShape(50.dp),
                color = DarkPrimary.copy(alpha = 0.2f)
            ) {
                Box(
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Lock",
                        tint = DarkPrimary,
                        modifier = Modifier.size(50.dp)
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Title
            Text(
                text = "Đổi mật khẩu",
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = DarkOnSurface
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Description
            Text(
                text = "Vui lòng nhập mật khẩu hiện tại và mật khẩu mới của bạn.",
                fontSize = 14.sp,
                color = DarkOnSurface.copy(alpha = 0.7f),
                textAlign = TextAlign.Center,
                lineHeight = 20.sp
            )
            
            Spacer(modifier = Modifier.height(32.dp))
            
            // Current Password
            OutlinedTextField(
                value = currentPassword,
                onValueChange = { 
                    currentPassword = it
                    errorMessage = null
                },
                label = { Text("Mật khẩu hiện tại") },
                leadingIcon = {
                    Icon(Icons.Default.Lock, "Mật khẩu hiện tại", tint = DarkPrimary)
                },
                trailingIcon = {
                    IconButton(onClick = { currentPasswordVisible = !currentPasswordVisible }) {
                        Text(
                            text = if (currentPasswordVisible) "👁" else "👁‍🗨",
                            fontSize = 20.sp,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                },
                visualTransformation = if (currentPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    focusedLabelColor = DarkPrimary,
                    unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                    cursorColor = DarkPrimary,
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                )
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // New Password
            OutlinedTextField(
                value = newPassword,
                onValueChange = { 
                    newPassword = it
                    errorMessage = null
                    passwordError = null
                },
                label = { Text("Mật khẩu mới") },
                leadingIcon = {
                    Icon(Icons.Default.Lock, "Mật khẩu mới", tint = DarkPrimary)
                },
                trailingIcon = {
                    IconButton(onClick = { newPasswordVisible = !newPasswordVisible }) {
                        Text(
                            text = if (newPasswordVisible) "👁" else "👁‍🗨",
                            fontSize = 20.sp,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                },
                visualTransformation = if (newPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    focusedLabelColor = DarkPrimary,
                    unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                    cursorColor = DarkPrimary,
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                ),
                isError = passwordError != null
            )
            
            if (passwordError != null) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = passwordError!!,
                    color = DarkError,
                    fontSize = 12.sp,
                    modifier = Modifier.fillMaxWidth()
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Confirm Password
            OutlinedTextField(
                value = confirmPassword,
                onValueChange = { 
                    confirmPassword = it
                    errorMessage = null
                    passwordError = null
                },
                label = { Text("Xác nhận mật khẩu mới") },
                leadingIcon = {
                    Icon(Icons.Default.Lock, "Xác nhận", tint = DarkPrimary)
                },
                trailingIcon = {
                    IconButton(onClick = { confirmPasswordVisible = !confirmPasswordVisible }) {
                        Text(
                            text = if (confirmPasswordVisible) "👁" else "👁‍🗨",
                            fontSize = 20.sp,
                            color = DarkOnSurface.copy(alpha = 0.6f)
                        )
                    }
                },
                visualTransformation = if (confirmPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = DarkPrimary,
                    unfocusedBorderColor = DarkOnSurface.copy(alpha = 0.3f),
                    focusedLabelColor = DarkPrimary,
                    unfocusedLabelColor = DarkOnSurface.copy(alpha = 0.6f),
                    cursorColor = DarkPrimary,
                    focusedTextColor = DarkOnSurface,
                    unfocusedTextColor = DarkOnSurface
                ),
                isError = passwordError != null
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            // Password requirements
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = DarkSurface.copy(alpha = 0.5f)
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                Column(
                    modifier = Modifier.padding(12.dp)
                ) {
                    Text(
                        "Yêu cầu mật khẩu:",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = DarkOnSurface.copy(alpha = 0.8f)
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "• Ít nhất 8 ký tự",
                        fontSize = 11.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                    Text(
                        "• Ít nhất 1 chữ hoa (A-Z)",
                        fontSize = 11.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                    Text(
                        "• Ít nhất 1 chữ thường (a-z)",
                        fontSize = 11.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                    Text(
                        "• Ít nhất 1 số (0-9)",
                        fontSize = 11.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                    Text(
                        "• Ít nhất 1 ký tự đặc biệt (@\$!%*?&)",
                        fontSize = 11.sp,
                        color = DarkOnSurface.copy(alpha = 0.6f)
                    )
                }
            }
            
            // Error message
            if (errorMessage != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = errorMessage!!,
                    color = DarkError,
                    fontSize = 14.sp,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }
            
            // Success message
            if (successMessage != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = DarkPrimary.copy(alpha = 0.2f)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Text(
                        text = "✓ $successMessage",
                        color = DarkPrimary,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(16.dp),
                        lineHeight = 20.sp
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(24.dp))
            
            // Submit button
            Button(
                onClick = {
                    // Validation
                    if (currentPassword.isBlank()) {
                        errorMessage = "Vui lòng nhập mật khẩu hiện tại"
                        return@Button
                    }
                    
                    if (newPassword.isBlank()) {
                        errorMessage = "Vui lòng nhập mật khẩu mới"
                        return@Button
                    }
                    
                    if (newPassword.length < 8) {
                        passwordError = "Mật khẩu phải có ít nhất 8 ký tự"
                        return@Button
                    }
                    
                    if (!newPassword.matches(Regex(".*[A-Z].*"))) {
                        passwordError = "Mật khẩu phải có ít nhất 1 chữ hoa"
                        return@Button
                    }
                    
                    if (!newPassword.matches(Regex(".*[a-z].*"))) {
                        passwordError = "Mật khẩu phải có ít nhất 1 chữ thường"
                        return@Button
                    }
                    
                    if (!newPassword.matches(Regex(".*\\d.*"))) {
                        passwordError = "Mật khẩu phải có ít nhất 1 số"
                        return@Button
                    }
                    
                    if (!newPassword.matches(Regex(".*[@\$!%*?&].*"))) {
                        passwordError = "Mật khẩu phải có ít nhất 1 ký tự đặc biệt (@\$!%*?&)"
                        return@Button
                    }
                    
                    if (newPassword != confirmPassword) {
                        passwordError = "Mật khẩu xác nhận không khớp"
                        return@Button
                    }
                    
                    if (newPassword == currentPassword) {
                        passwordError = "Mật khẩu mới không được trùng với mật khẩu hiện tại"
                        return@Button
                    }
                    
                    coroutineScope.launch {
                        isLoading = true
                        errorMessage = null
                        passwordError = null
                        
                        val result = changePasswordService.changePassword(
                            accessToken = accessToken,
                            currentPassword = currentPassword,
                            newPassword = newPassword
                        )
                        
                        isLoading = false
                        
                        result.onSuccess { response ->
                            successMessage = response.message
                            // Clear form
                            currentPassword = ""
                            newPassword = ""
                            confirmPassword = ""
                            
                            // Callback sau 2 giây
                            kotlinx.coroutines.delay(2000)
                            onSuccess()
                        }.onFailure { error ->
                            errorMessage = error.message
                        }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = !isLoading,
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = DarkPrimary,
                    disabledContainerColor = DarkPrimary.copy(alpha = 0.5f)
                )
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        color = DarkOnPrimary,
                        modifier = Modifier.size(24.dp)
                    )
                } else {
                    Text(
                        "Đổi mật khẩu",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}
