class RobotController {
    constructor() {
        this.baseTargetRotationY = 0;
        this.currentCommand = null;
        this.elbowTargetRotationX = 0;
        this.frameBuffer = [];
        this.fullCycleCount = 0;
        this.isBaseMoving = false;
        this.isBaseRotating = false;
        this.isElbowMoving = false;
        this.isElbowRotating = false;
        this.isRecording = false;
        this.isRecordingVideo = false;
        this.isShoulderMoving = false;
        this.isShoulderRotating = false;
        this.lastRecordedCommand = null;
        this.lastRecordedValue = null;
        this.maxElbowRotation = 45;
        this.maxShoulderRotation = 45;
        this.minElbowRotation = -90;
        this.minShoulderRotation = 0;
        this.movementStep = 0;
        this.recordedCommands = [];
        this.shoulderTargetRotationX = 0;
        this.targetBaseRotation = 0;
        this.targetElbowRotation = 0;
        this.targetShoulderRotation = 0;
    }

    automatedMovements() {
        switch (this.movementStep) {
            case 0:
                this.rotateBase(1);
                this.fullCycleCount += demoSpeed;
                if (this.fullCycleCount >= 2 * Math.PI) {
                    this.fullCycleCount = 0;
                    this.movementStep++;
                }
                break;
            default:
                this.movementStep = 0;
                break;
        }
    }

    checkAndResetAutomation() {
        if (this.targetBaseRotation === base.rotation.y &&
            this.targetShoulderRotation === shoulder.rotation.x &&
            this.targetElbowRotation === elbow.rotation.x) {
            this.isAutomatedCommandActive = false;
        }
    }

    clampRotation(currentRotation, minAngle, maxAngle) {
        const minRadians = THREE.MathUtils.degToRad(minAngle);
        const maxRadians = THREE.MathUtils.degToRad(maxAngle);
        return Math.max(minRadians, Math.min(maxRadians, currentRotation));
    }

    downloadRecordedCommands() {
        const text = this.recordedCommands.join('\n');
        const blob = new Blob([text], {
            type: 'text/plain'
        });
        const anchor = document.createElement('a');
        anchor.download = 'recorded_commands.txt';
        anchor.href = window.URL.createObjectURL(blob);
        anchor.click();
        window.URL.revokeObjectURL(anchor.href);
    }

    downloadVideo(videoBlob) {
        console.log("Preparing video for download...");
        const url = URL.createObjectURL(videoBlob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'recorded_video.mp4';
        anchor.click();
        window.URL.revokeObjectURL(url);
        console.log("Download should start now");
    }

    hasReachedTarget(joint, axis, targetRotation) {
        return Math.abs(joint.rotation[axis] - targetRotation) < 0.01;
    }

    interpolateRotation(joint, axis, targetRotation, deltaTime) {
        const currentRotation = joint.rotation[axis];
        const delta = targetRotation - currentRotation;

        deltaTime = Math.max(0, Math.min(deltaTime, 0.1));

        const rotationSpeed = 0.05;
        const movement = delta * deltaTime * rotationSpeed;

        if (Math.abs(delta) > 0.01) {
            joint.rotation[axis] += movement;
        } else {
            if (axis === 'y') {
                this.isBaseRotating = false;
            } else if (axis === 'x') {
                if (joint === shoulder) {
                    this.isShoulderRotating = false;
                } else if (joint === elbow) {
                    this.isElbowRotating = false;
                }
            }
        }
    }

    moveElbow(direction) {
        const previousRotation = elbow.rotation.x;
        const delta = demoSpeed * direction;
        elbow.rotation.x += delta;
        elbow.rotation.x = this.clampRotation(elbow.rotation.x, this.minElbowRotation, this.maxElbowRotation);
        if (this.isRecording) {
            this.recordCommand("moveElbow", elbow.rotation.x);
        }

        if (elbow.rotation.x !== previousRotation) {
            this.onMovementDetected();
        }
    }

    moveShoulder(direction) {
        const previousRotation = shoulder.rotation.x;
        const delta = demoSpeed * direction;
        shoulder.rotation.x += delta;
        shoulder.rotation.x = this.clampRotation(shoulder.rotation.x, this.minShoulderRotation, this.maxShoulderRotation);
        if (this.isRecording) {
            this.recordCommand("moveShoulder", shoulder.rotation.x);
        }

        if (shoulder.rotation.x !== previousRotation) {
            this.onMovementDetected();
        }
    }

    onMovementDetected() {
        const imageDataUrl = captureImage();
        updateForearmViewObjectDetection(imageDataUrl);
        processAndDisplayDepthMap();
    }

    radiansToDegrees(radians) {
        return (radians * 180 / Math.PI).toFixed(2);
    }

    recordCommand(command, value) {
        if (this.isRecording) {
            if (this.lastRecordedCommand !== command) {
                if (this.lastRecordedCommand !== null) {
                    this.recordedCommands.push(`${this.lastRecordedCommand} ${this.radiansToDegrees(this.lastRecordedValue)}`);
                }
                this.lastRecordedCommand = command;
                this.lastRecordedValue = value;
            } else {
                this.lastRecordedValue = value;
            }
        }
    }

    rotateBase(direction) {
        const previousRotation = base.rotation.y;
        const delta = demoSpeed * direction;
        base.rotation.y += delta;
        if (this.isRecording) {
            this.recordCommand("rotateBase", base.rotation.y);
        }

        if (base.rotation.y !== previousRotation) {
            this.onMovementDetected();
        }

        this.updateDirectionDisplay(base.rotation.y);
    }

    startRecording() {
        this.isRecording = true;
        this.recordedCommands = [];
        console.log("Recording started");
    }

    stopRecording() {
        if (this.isRecording) {
            if (this.lastRecordedCommand !== null) {
                this.recordedCommands.push(`${this.lastRecordedCommand} ${this.radiansToDegrees(this.lastRecordedValue)}`);
            }
            this.isRecording = false;
            this.lastRecordedCommand = null;
            this.lastRecordedValue = null;
            console.log("Recording stopped");
            console.log(this.recordedCommands);

            this.downloadRecordedCommands();
        }
    }

    startVideoRecording() {
        this.isRecordingVideo = true;
        console.log("Video recording started");

        const stream = forearmRenderer.domElement.captureStream(30);

        const options = {mimeType: 'video/webm'};
        this.mediaRecorder = new MediaRecorder(stream, options);

        this.recordedChunks = [];
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };

        this.mediaRecorder.start();
    }

    stopVideoRecording() {
        if (this.isRecordingVideo) {
            this.isRecordingVideo = false;
            console.log("Video recording stopped");

            this.mediaRecorder.stop();
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, {type: 'video/mp4'});
                console.log("Video compiled, starting download...");
                this.downloadVideo(blob);
            };
        }
    }

    update() {
        if (this.isBaseMoving) {
            this.interpolateRotation(base, 'y', this.targetBaseRotation);
            this.isBaseMoving = !this.hasReachedTarget(base, 'y', this.targetBaseRotation);
        }

        if (this.isBaseRotating) {
            this.interpolateRotation(base, 'y', this.baseTargetRotationY);
        }

        if (this.isElbowMoving) {
            this.interpolateRotation(elbow, 'x', this.targetElbowRotation);
            this.isElbowMoving = !this.hasReachedTarget(elbow, 'x', this.targetElbowRotation);
        }

        if (this.isElbowRotating) {
            this.interpolateRotation(elbow, 'x', this.elbowTargetRotationX);
        }

        if (this.isShoulderMoving) {
            this.interpolateRotation(shoulder, 'x', this.targetShoulderRotation);
            this.isShoulderMoving = !this.hasReachedTarget(shoulder, 'x', this.targetShoulderRotation);
        }

        if (this.isShoulderRotating) {
            this.interpolateRotation(shoulder, 'x', this.shoulderTargetRotationX);
        }
    }

    updateDirectionDisplay(rotation) {
        const degrees = THREE.Math.radToDeg(rotation);
        document.getElementById('directionValue').innerText = degrees.toFixed(2);
    }
}