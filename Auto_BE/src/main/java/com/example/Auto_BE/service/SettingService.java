package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.GetUserSettingsRequest;
import com.example.Auto_BE.dto.request.UpdateMultipleUserSettingsRequest;
import com.example.Auto_BE.dto.request.UpdateUserSettingRequest;
import com.example.Auto_BE.dto.response.AllUserSettingsResponse;
import com.example.Auto_BE.dto.response.SettingResponse;
import com.example.Auto_BE.dto.response.UserSettingResponse;
import com.example.Auto_BE.entity.SettingUser;
import com.example.Auto_BE.entity.Settings;
import com.example.Auto_BE.entity.User;
import com.example.Auto_BE.entity.enums.EUserType;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.SettingUserRepository;
import com.example.Auto_BE.repository.SettingsRepository;
import com.example.Auto_BE.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@Transactional
public class SettingService {

    @Autowired
    private SettingsRepository settingsRepository;
    
    @Autowired
    private SettingUserRepository settingUserRepository;
    
    @Autowired
    private UserRepository userRepository;

    /**
     * Lấy tất cả các loại settings có sẵn (danh sách settings)
     */
    public BaseResponse<List<SettingResponse>> getAllSettings() {
        try {
            List<Settings> settings = settingsRepository.findByIsActiveTrueOrderBySettingKeyAsc();
            
            List<SettingResponse> responseList = settings.stream()
                    .map(this::convertToSettingResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<SettingResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách settings thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            System.err.println("Error getting all settings: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<List<SettingResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách settings: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Lấy tất cả giá trị settings của user
     * Nếu userId = null thì là GUEST, trả về giá trị mặc định
     * Nếu userId != null thì query từ bảng SettingUser
     */
    public BaseResponse<AllUserSettingsResponse> getUserSettings(GetUserSettingsRequest request) {
        try {
            User user = null;
            EUserType userType = EUserType.GUEST;
            
            // Xác định user và userType từ userId
            if (request.getUserId() != null) {
                Optional<User> userOpt = userRepository.findById(request.getUserId());
                if (userOpt.isPresent()) {
                    user = userOpt.get();
                    // Xác định userType dựa vào instance type
                    if (user instanceof com.example.Auto_BE.entity.ElderUser) {
                        userType = EUserType.ELDER;
                    } else if (user instanceof com.example.Auto_BE.entity.SupervisorUser) {
                        userType = EUserType.SUPERVISOR;
                    }
                } else {
                    return BaseResponse.<AllUserSettingsResponse>builder()
                            .status("error")
                            .message("User không tồn tại với ID: " + request.getUserId())
                            .data(null)
                            .build();
                }
            }

            // Lấy tất cả settings active
            List<Settings> allSettings = settingsRepository.findByIsActiveTrueOrderBySettingKeyAsc();
            
            // Lấy settings của user (nếu có)
            List<SettingUser> userSettings = new ArrayList<>();
            if (user != null) {
                userSettings = settingUserRepository.findByUserOrderBySetting_SettingKeyAsc(user);
            }

            // Tạo map để tra cứu nhanh
            java.util.Map<String, String> userSettingMap = userSettings.stream()
                    .collect(Collectors.toMap(
                            su -> su.getSetting().getSettingKey(),
                            SettingUser::getValue
                    ));

            // Tạo response: nếu user có setting thì dùng giá trị của user, không thì dùng default
            List<UserSettingResponse> settingsResponse = allSettings.stream()
                    .map(setting -> {
                        String value = userSettingMap.getOrDefault(
                                setting.getSettingKey(),
                                setting.getDefaultValue()
                        );
                        return UserSettingResponse.builder()
                                .settingKey(setting.getSettingKey())
                                .value(value)
                                .defaultValue(setting.getDefaultValue())
                                .build();
                    })
                    .collect(Collectors.toList());

            AllUserSettingsResponse response = AllUserSettingsResponse.builder()
                    .userId(user != null ? user.getId() : null)
                    .userType(userType.toString())
                    .settings(settingsResponse)
                    .build();

            return BaseResponse.<AllUserSettingsResponse>builder()
                    .status("success")
                    .message("Lấy settings của user thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error getting user settings: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<AllUserSettingsResponse>builder()
                    .status("error")
                    .message("Lỗi khi lấy settings của user: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Cập nhật 1 setting của user
     * Nếu userId = null thì không thể lưu (GUEST không lưu được)
     */
    public BaseResponse<UserSettingResponse> updateUserSetting(UpdateUserSettingRequest request) {
        try {
            // Nếu userId = null thì không thể lưu
            if (request.getUserId() == null) {
                return BaseResponse.<UserSettingResponse>builder()
                        .status("error")
                        .message("Cần đăng nhập để lưu setting")
                        .data(null)
                        .build();
            }

            // Tìm user
            User user = userRepository.findById(request.getUserId())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại với ID: " + request.getUserId()));

            // Tìm setting
            Settings setting = settingsRepository.findBySettingKey(request.getSettingKey())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "Setting không tồn tại: " + request.getSettingKey()));

            // Validate value (kiểm tra value có trong possibleValues không)
            if (setting.getPossibleValues() != null && !setting.getPossibleValues().isEmpty()) {
                String[] possibleValues = setting.getPossibleValues().split(",");
                boolean isValid = false;
                for (String possibleValue : possibleValues) {
                    if (possibleValue.trim().equalsIgnoreCase(request.getValue())) {
                        isValid = true;
                        break;
                    }
                }
                if (!isValid) {
                    return BaseResponse.<UserSettingResponse>builder()
                            .status("error")
                            .message("Giá trị không hợp lệ. Các giá trị có thể: " + setting.getPossibleValues())
                            .data(null)
                            .build();
                }
            }

            // Tìm hoặc tạo SettingUser
            Optional<SettingUser> settingUserOpt = settingUserRepository.findBySettingAndUser(setting, user);
            SettingUser settingUser;
            
            if (settingUserOpt.isPresent()) {
                // Update existing
                settingUser = settingUserOpt.get();
                settingUser.setValue(request.getValue());
            } else {
                // Create new
                settingUser = new SettingUser();
                settingUser.setSetting(setting);
                settingUser.setUser(user);
                settingUser.setValue(request.getValue());
            }

            // Save
            SettingUser saved = settingUserRepository.save(settingUser);

            // Convert to response
            UserSettingResponse response = UserSettingResponse.builder()
                    .settingKey(saved.getSetting().getSettingKey())
                    .value(saved.getValue())
                    .defaultValue(saved.getSetting().getDefaultValue())
                    .build();

            return BaseResponse.<UserSettingResponse>builder()
                    .status("success")
                    .message("Cập nhật setting thành công")
                    .data(response)
                    .build();

        } catch (Exception e) {
            System.err.println("Error updating user setting: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<UserSettingResponse>builder()
                    .status("error")
                    .message("Lỗi khi cập nhật setting: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Cập nhật nhiều settings của user cùng lúc
     * Nếu userId = null thì không thể lưu (GUEST không lưu được)
     */
    public BaseResponse<AllUserSettingsResponse> updateMultipleUserSettings(UpdateMultipleUserSettingsRequest request) {
        try {
            // Nếu userId = null thì không thể lưu
            if (request.getUserId() == null) {
                return BaseResponse.<AllUserSettingsResponse>builder()
                        .status("error")
                        .message("Cần đăng nhập để lưu settings")
                        .data(null)
                        .build();
            }

            // Tìm user
            User user = userRepository.findById(request.getUserId())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại với ID: " + request.getUserId()));

            // Cập nhật từng setting (dùng userId từ request chính, không dùng từ settingRequest)
            List<String> errors = new ArrayList<>();
            for (UpdateUserSettingRequest settingRequest : request.getSettings()) {
                try {
                    // Tìm setting
                    Optional<Settings> settingOpt = settingsRepository.findBySettingKey(settingRequest.getSettingKey());
                    if (!settingOpt.isPresent()) {
                        errors.add("Setting không tồn tại: " + settingRequest.getSettingKey());
                        continue;
                    }
                    
                    Settings setting = settingOpt.get();
                    
                    // Validate value
                    if (setting.getPossibleValues() != null && !setting.getPossibleValues().isEmpty()) {
                        String[] possibleValues = setting.getPossibleValues().split(",");
                        boolean isValid = false;
                        for (String possibleValue : possibleValues) {
                            if (possibleValue.trim().equalsIgnoreCase(settingRequest.getValue())) {
                                isValid = true;
                                break;
                            }
                        }
                        if (!isValid) {
                            errors.add("Giá trị không hợp lệ cho " + settingRequest.getSettingKey() + 
                                    ": " + settingRequest.getValue());
                            continue;
                        }
                    }

                    // Tìm hoặc tạo SettingUser (dùng user từ request chính)
                    Optional<SettingUser> settingUserOpt = settingUserRepository.findBySettingAndUser(setting, user);
                    SettingUser settingUser;
                    
                    if (settingUserOpt.isPresent()) {
                        settingUser = settingUserOpt.get();
                        settingUser.setValue(settingRequest.getValue());
                    } else {
                        settingUser = new SettingUser();
                        settingUser.setSetting(setting);
                        settingUser.setUser(user);
                        settingUser.setValue(settingRequest.getValue());
                    }
                    
                    settingUserRepository.save(settingUser);
                    
                } catch (Exception e) {
                    errors.add("Lỗi khi cập nhật " + settingRequest.getSettingKey() + ": " + e.getMessage());
                }
            }

            if (!errors.isEmpty()) {
                return BaseResponse.<AllUserSettingsResponse>builder()
                        .status("error")
                        .message("Có lỗi xảy ra: " + String.join(", ", errors))
                        .data(null)
                        .build();
            }

            // Lấy lại tất cả settings sau khi update
            GetUserSettingsRequest getUserSettingsRequest = GetUserSettingsRequest.builder()
                    .userId(request.getUserId())
                    .build();
            return getUserSettings(getUserSettingsRequest);

        } catch (Exception e) {
            System.err.println("Error updating multiple user settings: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<AllUserSettingsResponse>builder()
                    .status("error")
                    .message("Lỗi khi cập nhật settings: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Reset settings của user về giá trị mặc định (xóa tất cả custom settings)
     */
    public BaseResponse<String> resetUserSettingsToDefault(Long userId) {
        try {
            if (userId == null) {
                return BaseResponse.<String>builder()
                        .status("error")
                        .message("Cần đăng nhập để reset settings")
                        .data(null)
                        .build();
            }

            // Tìm user
            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại với ID: " + userId));

            // Xóa tất cả settings của user
            settingUserRepository.deleteByUser(user);

            return BaseResponse.<String>builder()
                    .status("success")
                    .message("Đã reset settings về giá trị mặc định")
                    .data("Settings đã được reset")
                    .build();

        } catch (Exception e) {
            System.err.println("Error resetting user settings: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<String>builder()
                    .status("error")
                    .message("Lỗi khi reset settings: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    /**
     * Khởi tạo settings mặc định cho user mới (gọi khi đăng ký thành công)
     * Tạo records trong SettingUser với giá trị default từ Settings
     */
    public void initializeUserSettings(User user) {
        try {
            // Lấy tất cả settings active
            List<Settings> allSettings = settingsRepository.findByIsActiveTrueOrderBySettingKeyAsc();
            
            // Tạo SettingUser cho mỗi setting với giá trị mặc định
            for (Settings setting : allSettings) {
                // Kiểm tra xem đã có chưa (tránh duplicate)
                if (!settingUserRepository.existsBySettingAndUser(setting, user)) {
                    SettingUser settingUser = new SettingUser();
                    settingUser.setSetting(setting);
                    settingUser.setUser(user);
                    settingUser.setValue(setting.getDefaultValue());
                    settingUserRepository.save(settingUser);
                }
            }
            
            System.out.println("Initialized settings for user ID: " + user.getId());
        } catch (Exception e) {
            System.err.println("Error initializing user settings: " + e.getMessage());
            e.printStackTrace();
            // Không throw exception để không ảnh hưởng đến quá trình đăng ký
        }
    }

    // ===== HELPER METHODS =====

    private SettingResponse convertToSettingResponse(Settings setting) {
        return SettingResponse.builder()
                .id(setting.getId())
                .settingKey(setting.getSettingKey())
                .name(setting.getName())
                .description(setting.getDescription())
                .defaultValue(setting.getDefaultValue())
                .possibleValues(setting.getPossibleValues())
                .isActive(setting.getIsActive())
                .build();
    }
}
