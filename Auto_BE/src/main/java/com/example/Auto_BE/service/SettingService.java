package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.BaseResponse;
import com.example.Auto_BE.dto.request.UpdateMultipleUserSettingsRequest;
import com.example.Auto_BE.dto.request.UpdateUserSettingRequest;
import com.example.Auto_BE.dto.response.AllUserSettingsResponse;
import com.example.Auto_BE.dto.response.SettingResponse;
import com.example.Auto_BE.dto.response.UserSettingResponse;
import com.example.Auto_BE.entity.*;
import com.example.Auto_BE.entity.enums.ESettingType;
import com.example.Auto_BE.entity.enums.EUserType;
import com.example.Auto_BE.exception.BaseException;
import com.example.Auto_BE.repository.SettingUserRepository;
import com.example.Auto_BE.repository.SettingsRepository;
import com.example.Auto_BE.repository.UserRepository;
import org.springframework.security.core.Authentication;
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

    public BaseResponse<List<SettingResponse>> getAllBySettingType(ESettingType settingType) {
        try {
            List<Settings> settings = settingsRepository.findBySettingTypeAndIsActiveTrueOrderBySettingKeyAsc(settingType);
            
            List<SettingResponse> responseList = settings.stream()
                    .map(this::convertToSettingResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<SettingResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách settings thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            System.err.println("Error getting settings by type: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<List<SettingResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách settings: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    public BaseResponse<List<SettingResponse>> getAllBySettingTypes(List<ESettingType> settingTypes) {
        try {
            List<Settings> settings = settingsRepository.findBySettingTypeInAndIsActiveTrueOrderBySettingKeyAsc(settingTypes);
            
            List<SettingResponse> responseList = settings.stream()
                    .map(this::convertToSettingResponse)
                    .collect(Collectors.toList());

            return BaseResponse.<List<SettingResponse>>builder()
                    .status("success")
                    .message("Lấy danh sách settings thành công")
                    .data(responseList)
                    .build();

        } catch (Exception e) {
            System.err.println("Error getting settings by types: " + e.getMessage());
            e.printStackTrace();
            return BaseResponse.<List<SettingResponse>>builder()
                    .status("error")
                    .message("Lỗi khi lấy danh sách settings: " + e.getMessage())
                    .data(null)
                    .build();
        }
    }

    public BaseResponse<AllUserSettingsResponse> getUserSettings(Authentication authentication) {
        try {
            User user = null;
            EUserType userType = null;
            
            if (authentication != null && authentication.getName() != null) {
                Optional<User> userOpt = userRepository.findByEmail(authentication.getName());
                if (userOpt.isPresent()) {
                    user = userOpt.get();
                    if (user instanceof ElderUser) {
                        userType = EUserType.ELDER;
                    } else if (user instanceof SupervisorUser) {
                        userType = EUserType.SUPERVISOR;
                    }
                }
            }

            List<Settings> allSettings = settingsRepository.findByIsActiveTrueOrderBySettingKeyAsc();
            
            List<SettingUser> userSettings = new ArrayList<>();
            if (user != null) {
                userSettings = settingUserRepository.findByUserOrderBySetting_SettingKeyAsc(user);
            }

            java.util.Map<String, String> userSettingMap = userSettings.stream()
                    .collect(Collectors.toMap(
                            su -> su.getSetting().getSettingKey(),
                            SettingUser::getValue
                    ));

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
                    .userType(userType != null ? userType.toString() : null)
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

    public BaseResponse<UserSettingResponse> updateUserSetting(UpdateUserSettingRequest request, Authentication authentication) {
        try {
            if (authentication == null || authentication.getName() == null) {
                return BaseResponse.<UserSettingResponse>builder()
                        .status("error")
                        .message("Cần đăng nhập để lưu setting")
                        .data(null)
                        .build();
            }

            User user = userRepository.findByEmail(authentication.getName())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại"));

            Settings setting = settingsRepository.findBySettingKey(request.getSettingKey())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "Setting không tồn tại: " + request.getSettingKey()));

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

            Optional<SettingUser> settingUserOpt = settingUserRepository.findBySettingAndUser(setting, user);
            SettingUser settingUser;
            
            if (settingUserOpt.isPresent()) {
                settingUser = settingUserOpt.get();
                settingUser.setValue(request.getValue());
            } else {
                settingUser = new SettingUser();
                settingUser.setSetting(setting);
                settingUser.setUser(user);
                settingUser.setValue(request.getValue());
            }

            SettingUser saved = settingUserRepository.save(settingUser);

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

    public BaseResponse<AllUserSettingsResponse> updateMultipleUserSettings(UpdateMultipleUserSettingsRequest request) {
        try {
            if (request.getUserId() == null) {
                return BaseResponse.<AllUserSettingsResponse>builder()
                        .status("error")
                        .message("Cần đăng nhập để lưu settings")
                        .data(null)
                        .build();
            }

            User user = userRepository.findById(request.getUserId())
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại với ID: " + request.getUserId()));

            List<String> errors = new ArrayList<>();
            for (UpdateUserSettingRequest settingRequest : request.getSettings()) {
                try {
                    Optional<Settings> settingOpt = settingsRepository.findBySettingKey(settingRequest.getSettingKey());
                    if (!settingOpt.isPresent()) {
                        errors.add("Setting không tồn tại: " + settingRequest.getSettingKey());
                        continue;
                    }
                    
                    Settings setting = settingOpt.get();
                    
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

            org.springframework.security.authentication.UsernamePasswordAuthenticationToken auth = 
                    new org.springframework.security.authentication.UsernamePasswordAuthenticationToken(
                            user.getEmail(), null, null);
            return getUserSettings(auth);

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

    public BaseResponse<String> resetUserSettingsToDefault(Long userId) {
        try {
            if (userId == null) {
                return BaseResponse.<String>builder()
                        .status("error")
                        .message("Cần đăng nhập để reset settings")
                        .data(null)
                        .build();
            }

            User user = userRepository.findById(userId)
                    .orElseThrow(() -> new BaseException.EntityNotFoundException(
                            "User không tồn tại với ID: " + userId));

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

    public void initializeUserSettings(User user) {
        try {
            List<Settings> allSettings = settingsRepository.findByIsActiveTrueOrderBySettingKeyAsc();
            
            for (Settings setting : allSettings) {
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
        }
    }

    private SettingResponse convertToSettingResponse(Settings setting) {
        return SettingResponse.builder()
                .id(setting.getId())
                .settingKey(setting.getSettingKey())
                .name(setting.getName())
                .description(setting.getDescription())
                .defaultValue(setting.getDefaultValue())
                .possibleValues(setting.getPossibleValues())
                .isActive(setting.getIsActive())
                .settingType(setting.getSettingType())
                .build();
    }
}
