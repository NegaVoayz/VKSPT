#pragma once

/// Camera parameters passed each frame: origin + three basis vectors
/// (U = right*tanFovX, V = up*tanFovY, W = forward).
struct CameraParams {
    float origin[3];
    float camU[3];
    float camV[3];
    float camW[3];
};
