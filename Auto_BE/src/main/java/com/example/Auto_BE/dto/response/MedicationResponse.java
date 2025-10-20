package com.example.Auto_BE.dto.response;

import com.example.Auto_BE.entity.enums.ETypeMedication;
import lombok.*;

import java.time.LocalDateTime;
import java.util.List;

/**
 * 💊 UNIFIED Response cho Medication
 * 
 * ✅ Đồng bộ với CreateMedicationRequest
 * ✅ Hỗ trợ multiple reminder times
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MedicationResponse {

    private Long id;
    private Long userId;
    private String userName;
    private Long prescriptionId;
    
    // ✅ UNIFIED FIELDS
    private String medicationName;
    private ETypeMedication type;                  // ✅ Loại thuốc
    private List<String> reminderTimes;            // ✅ Array: ["08:00", "14:00", "20:00"]
    private String daysOfWeek;                     // ✅ "1111111"
    private String notes;
    
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    // Backward compatibility (deprecated)
    @Deprecated
    public String getReminderTime() {
        return reminderTimes != null && !reminderTimes.isEmpty() 
            ? reminderTimes.get(0) 
            : null;
    }
}