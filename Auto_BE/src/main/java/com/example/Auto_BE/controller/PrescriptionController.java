//package com.example.Auto_BE.controller;
//
//import com.example.Auto_BE.dto.BaseResponse;
//import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
//import com.example.Auto_BE.dto.response.PrescriptionResponse;
//import com.example.Auto_BE.entity.BaseEntity;
//import com.example.Auto_BE.service.PrescriptionService;
//import jakarta.validation.Valid;
//import org.springframework.http.ResponseEntity;
//import org.springframework.security.core.Authentication;
//import org.springframework.web.bind.annotation.*;
//
//@RestController
//@RequestMapping("/api/prescriptions")
//public class PrescriptionController {
//    private final PrescriptionService prescriptionService;
//
//    public PrescriptionController(PrescriptionService prescriptionService) {
//        this.prescriptionService = prescriptionService;
//    }
//
//    @PostMapping("/create")
//    public ResponseEntity<BaseResponse<PrescriptionResponse>> createPrescription(@RequestBody @Valid PrescriptionCreateRequest prescriptionCreateRequest,
//                                                                                     Authentication authentication) {
//        BaseResponse<PrescriptionResponse> response = prescriptionService.create(prescriptionCreateRequest, authentication);
//        return ResponseEntity.ok(response);
//    }
//
//
//
//
//}
