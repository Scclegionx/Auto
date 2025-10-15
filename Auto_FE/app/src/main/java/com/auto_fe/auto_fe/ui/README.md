# UI Architecture - Auto FE

## Cấu trúc thư mục UI mới

```
ui/
├── screens/           # Các màn hình chính
│   ├── VoiceScreen.kt      # Màn hình ghi âm (mặc định)
│   ├── MedicineScreen.kt   # Màn hình quản lý thuốc
│   └── SettingsScreen.kt   # Màn hình cài đặt
├── components/        # Các component UI tái sử dụng
│   ├── BottomNavigation.kt     # Bottom Navigation với 3 nút
│   ├── SoftControlButtons.kt   # Nút điều khiển mềm
│   └── Rotating3DSphere.kt     # Sphere 3D xoay
├── service/           # Service layer cho API calls
│   └── ApiService.kt          # Gọi API Backend
└── theme/             # Theme và styling
    ├── Color.kt
    ├── Theme.kt
    └── Typography.kt
```

## Tính năng chính

### 1. Bottom Navigation
- **3 nút chính:**
  - 💊 **Thuốc** (trái) - Quản lý lịch uống thuốc
  - 🎤 **Ghi âm** (giữa) - Màn hình chính, style đặc biệt
  - ⚙️ **Cài đặt** (phải) - Cài đặt ứng dụng

- **Nút ghi âm đặc biệt:**
  - Style khác biệt và nổi bật hơn
  - Animation đặc biệt khi được chọn
  - Là màn hình mặc định khi mở app

### 2. Màn hình ghi âm (VoiceScreen)
- **Màn hình chính** - mặc định khi mở app
- Tích hợp đầy đủ với VoiceManager hiện có
- Animation mượt mà với voice level
- Background 3D sphere xoay
- Control buttons với haptic feedback

### 3. Màn hình thuốc (MedicineScreen)
- Hiển thị danh sách thuốc
- Quản lý lịch uống thuốc
- Status tracking (Đang dùng, Hết thuốc, Tạm dừng)
- Nút thêm thuốc mới

### 4. Màn hình cài đặt (SettingsScreen)
- **Âm thanh & Giọng nói:**
  - Bật/tắt trợ lý giọng nói
  - Thông báo âm thanh
- **Ứng dụng:**
  - Tự động khởi động
  - Ngôn ngữ (Tiếng Việt, English, 中文)
  - Giao diện (Tối, Sáng, Tự động)
- **Thông tin:**
  - Phiên bản
  - Nhà phát triển
  - Đánh giá ứng dụng
  - Chia sẻ ứng dụng

### 5. Service Layer (ApiService)
- **Tách biệt logic API** khỏi UI components
- **Các API chính:**
  - `sendVoiceData()` - Gửi dữ liệu giọng nói
  - `getMedicines()` - Lấy danh sách thuốc
  - `updateSettings()` - Cập nhật cài đặt
  - `sendFeedback()` - Gửi feedback
- **Error handling** và **logging** đầy đủ

## Cách sử dụng

### 1. MainActivity
```kotlin
@Composable
fun MainScreen() {
    var selectedTab by remember { mutableStateOf(1) } // Default là tab ghi âm
    
    Scaffold(
        bottomBar = {
            CustomBottomNavigation(
                selectedTab = selectedTab,
                onTabSelected = { selectedTab = it }
            )
        }
    ) { innerPadding ->
        when (selectedTab) {
            0 -> MedicineScreen()
            1 -> VoiceScreen() // Màn hình mặc định
            2 -> SettingsScreen()
        }
    }
}
```

### 2. Sử dụng ApiService
```kotlin
val apiService = ApiService(context)

// Gửi dữ liệu giọng nói
val result = apiService.sendVoiceData(audioData, transcript, userId)
when (result) {
    is ApiResult.Success -> { /* Handle success */ }
    is ApiResult.Error -> { /* Handle error */ }
}
```

## Lợi ích của cấu trúc mới

1. **Tách biệt rõ ràng:** UI, logic, và API calls được tách biệt
2. **Dễ maintain:** Mỗi màn hình có file riêng
3. **Tái sử dụng:** Components có thể dùng lại
4. **Scalable:** Dễ thêm màn hình mới
5. **Clean Architecture:** Service layer tách biệt API logic
6. **User Experience:** Bottom navigation với màn hình ghi âm làm chính

## Jetpack Compose

Tất cả UI được xây dựng bằng **Jetpack Compose** với:
- Material 3 Design System
- Animation mượt mà
- Dark theme mặc định
- Responsive design
- Haptic feedback
