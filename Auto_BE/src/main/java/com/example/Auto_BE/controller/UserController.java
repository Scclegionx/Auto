package com.example.Auto_BE.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/users")
public class UserController {
    @GetMapping
    public String getUsers() {
        // This is a placeholder method. You can implement your logic to fetch users here.
        return "List of users";
    }
}
