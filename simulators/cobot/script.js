const img = document.getElementById('img');
const robot = new RobotController();

function addFloor() {
    const backgroundGeometry = new THREE.PlaneGeometry(10, 10);

    const backgroundMaterial = new THREE.MeshStandardMaterial({
        color: 0x000000,
        side: THREE.DoubleSide,
    });

    const background = new THREE.Mesh(backgroundGeometry, backgroundMaterial);
    background.rotation.x = -Math.PI / 2;
    background.position.y = -0.01;
    scene.add(background);

    /*const gridGeometry = new THREE.PlaneGeometry(10, 10, 10, 10);
    const gridMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5,
        wireframe: true,
    });
    const grid = new THREE.Mesh(gridGeometry, gridMaterial);
    grid.rotation.x = -Math.PI / 2;
    grid.position.y = 0.001;
    scene.add(grid);*/

    var floorShape = new CANNON.Plane();
    var floorBody = new CANNON.Body({
        mass: 0
    });
    floorBody.addShape(floorShape);
    floorBody.quaternion.setFromEuler(-Math.PI / 2, 0, 0);
    world.addBody(floorBody);

    /*const floorVisualGeometry = new THREE.PlaneGeometry(10, 10);
    const floorVisualMaterial = new THREE.MeshBasicMaterial({
        color: 0x00ff00,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5,
        wireframe: true
    });
    const floorVisual = new THREE.Mesh(floorVisualGeometry, floorVisualMaterial);
    floorVisual.rotation.x = -Math.PI / 2;
    floorVisual.position.y = -0.01;
    scene.add(floorVisual);*/
}

function addLighting() {
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 10, 10).normalize();
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.5, 100);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
}

function addPhysicsObject(type, size = {
    x: 0.5,
    y: 0.5,
    z: 0.5
}, position = {
    x: 0,
    y: 0.25,
    z: 1.25
}) {
    let geometry;
    let meshMaterial = new THREE.MeshNormalMaterial();

    switch (type) {
        case 'box':
            geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
            break;
        case 'sphere':
            geometry = new THREE.SphereGeometry(size.x, 32, 32);
            break;
        default:
            console.warn('Unsupported physics object type:', type);
            return;
    }

    const mesh = new THREE.Mesh(geometry, meshMaterial);
    mesh.position.set(position.x, position.y, position.z);
    scene.add(mesh);

    pairMeshWithPhysicsBody(mesh, {
        mass: 10,
        friction: 0.4,
        restitution: 0.6,
        isKinematic: false
    });
}

function addRobotArm() {
    // Material for the robot parts
    var robotMaterial = new THREE.MeshNormalMaterial();

    // Base of the robot
    const baseGeometry = new THREE.CylinderGeometry(0.2, 0.2, 0.2, 32);
    base = new THREE.Mesh(baseGeometry, robotMaterial);
    base.position.y = 0.1;
    scene.add(base);

    // Shoulder of the robot
    const shoulderGeometry = new THREE.SphereGeometry(0.2, 32, 32);
    shoulder = new THREE.Mesh(shoulderGeometry, robotMaterial);
    shoulder.position.y = 0.1;
    shoulder.rotation.x = THREE.MathUtils.degToRad(0);
    base.add(shoulder);

    // Upper arm of the robot
    const upperArmGeometry = new THREE.BoxGeometry(0.1, 1, 0.1);
    upperArm = new THREE.Mesh(upperArmGeometry, robotMaterial);
    upperArm.position.y = 0.5;
    shoulder.add(upperArm);
    pairMeshWithPhysicsBody(upperArm, {
        mass: 0, friction: 0.4, restitution: 0.6, isKinematic: true
    });

    // Elbow of the robot
    const elbowGeometry = new THREE.CylinderGeometry(0.15, 0.15, 0.15, 32);
    elbow = new THREE.Mesh(elbowGeometry, robotMaterial);
    elbow.position.y = 0.6;
    elbow.rotation.z = Math.PI / 2;
    elbow.rotation.x = THREE.MathUtils.degToRad(45);
    elbow.castShadow = true;
    elbow.receiveShadow = true;
    upperArm.add(elbow);

    // Forearm of the robot
    const forearmGeometry = new THREE.BoxGeometry(0.1, 0.1, 0.7);
    forearm = new THREE.Mesh(forearmGeometry, robotMaterial);
    forearm.position.set(0, 0, 0.4);
    elbow.add(forearm);
    pairMeshWithPhysicsBody(forearm, {
        mass: 0, friction: 0.4, restitution: 0.6, isKinematic: true
    });

    // Pincer base and claws of the robot
    const pincerBaseGeometry = new THREE.BoxGeometry(0.2, 0.4, 0.05);
    pincerBase = new THREE.Mesh(pincerBaseGeometry, robotMaterial);
    forearm.add(pincerBase);
    pincerBase.position.set(0, 0, 0.36);
    pairMeshWithPhysicsBody(pincerBase, {
        mass: 0, friction: 0.4, restitution: 0.6, isKinematic: true
    });

    const pincerClawGeometry = new THREE.BoxGeometry(0.2, 0.05, 0.1);
    pincerClaw1 = new THREE.Mesh(pincerClawGeometry, robotMaterial);
    pincerClaw2 = new THREE.Mesh(pincerClawGeometry, robotMaterial);
    pincerClaw1.position.set(0, 0.175, 0.075);
    pincerClaw2.position.set(0, -0.175, 0.075);
    pincerBase.add(pincerClaw1);
    pincerBase.add(pincerClaw2);
    pairMeshWithPhysicsBody(pincerClaw1, {
        mass: 0, friction: 0.4, restitution: 0.6, isKinematic: true
    });
    pairMeshWithPhysicsBody(pincerClaw2, {
        mass: 0, friction: 0.4, restitution: 0.6, isKinematic: true
    });
}

function animate() {
    requestAnimationFrame(animate);

    if (world) {
        world.step(1 / 60);

        scene.traverse(function (object) {
            if (object.userData.physicsBody) {
                if (object.userData.physicsBody.type === CANNON.Body.DYNAMIC) {
                    object.position.copy(object.userData.physicsBody.position);
                    object.quaternion.copy(object.userData.physicsBody.quaternion);
                } else if (object.userData.physicsBody.type === CANNON.Body.KINEMATIC) {
                    var worldPosition = new THREE.Vector3();
                    var worldQuaternion = new THREE.Quaternion();
                    object.getWorldPosition(worldPosition);
                    object.getWorldQuaternion(worldQuaternion);
                    object.userData.physicsBody.position.copy(worldPosition);
                    object.userData.physicsBody.quaternion.copy(worldQuaternion);
                }
            }
        });
    }

    robot.update();
    //cannonDebugRenderer.update();
    renderer.render(scene, camera);

    if (forearmCameraIsActive) {
        updateForearmCamera();
        forearmRenderer.render(scene, forearmCamera);
    }
}

function automatedMovements() {
    switch (this.movementStep) {
        case 0:
            this.rotateBase(1);
            this.onMovementDetected();
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

function clampRotation(currentRotation, minAngle, maxAngle) {
    const minRadians = THREE.MathUtils.degToRad(minAngle);
    const maxRadians = THREE.MathUtils.degToRad(maxAngle);
    return Math.max(minRadians, Math.min(maxRadians, currentRotation));
}

function handleRecordButton() {
    if (robot.isRecording) {
        robot.stopRecording();
        robot.stopVideoRecording();
        this.innerText = "Record";
    } else {
        robot.startRecording();
        robot.startVideoRecording();
        this.innerText = "Stop";
    }
}

function handleResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(onWindowResize, 250);
}

function initForearmCamera() {
    let width = 512;
    let height = 512;
    const aspect = width / height;
    const fov = 60;
    forearmCamera = new THREE.PerspectiveCamera(fov, aspect, 0.1, 1000);
    forearmRenderer = new THREE.WebGLRenderer({
        antialias: true
    });

    forearmRenderer.setSize(width, height);
    forearmRenderer.gammaFactor = 2.2;
    forearmRenderer.outputEncoding = THREE.sRGBEncoding;
    forearmRenderer.shadowMap.enabled = true;
    forearmRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
    forearmRenderer.setClearColor(0x000000);
    document.getElementById('forearmViewObjectDetection').style.width = `128px`;
    document.getElementById('forearmViewObjectDetection').style.height = `128px`;
    document.getElementById('forearmView').appendChild(forearmRenderer.domElement);
}

function initRobot(type) {
    world = new CANNON.World();
    world.gravity.set(0, -9.82, 0);

    setupScene();
    addFloor();
    addLighting();
    setupCamera();
    setupRenderer();

    let cannonDebugRenderer = new THREE.CannonDebugRenderer(scene, world);

    switch (type) {
        case 'cobot':

            addPhysicsObject('box', {
                x: 0.4,
                y: 0.4,
                z: 0.4
            }, {
                x: 0,
                y: 0.2,
                z: 1.20
            });

            addRobotArm();
            initForearmCamera();
            setupControls();

            const initialImage = captureImage();
            updateForearmViewObjectDetection(initialImage);
            break;
    }

    animate();
    //processAndDisplayDepthMap();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);

    const forearmAspect = 512 / 512;
    forearmCamera.aspect = forearmAspect;
    forearmCamera.updateProjectionMatrix();
    forearmRenderer.setSize(512 * window.devicePixelRatio, 512 * window.devicePixelRatio);
}

function pairMeshWithPhysicsBody(mesh, options = {}) {
    let shape;
    const geometry = mesh.geometry;
    const mass = options.mass !== undefined ? options.mass : 1;
    const isKinematic = options.isKinematic || false;

    if (geometry instanceof THREE.BoxGeometry) {
        const size = new CANNON.Vec3(geometry.parameters.width / 2, geometry.parameters.height / 2, geometry.parameters.depth / 2);
        shape = new CANNON.Box(size);
    } else if (geometry instanceof THREE.SphereGeometry) {
        const radius = geometry.parameters.radius;
        shape = new CANNON.Sphere(radius);
    }

    const body = new CANNON.Body({
        mass: isKinematic ? 0 : mass,
        material: new CANNON.Material({
            friction: options.friction || 0.3,
            restitution: options.restitution || 0.5,
        }),
        shape: shape,
        type: isKinematic ? CANNON.Body.KINEMATIC : CANNON.Body.DYNAMIC
    });

    body.position.set(mesh.position.x, mesh.position.y, mesh.position.z);
    body.quaternion.set(mesh.quaternion.x, mesh.quaternion.y, mesh.quaternion.z, mesh.quaternion.w);

    world.addBody(body);
    mesh.userData.physicsBody = body;
}

function parseCommands(commandsArray) {
    if (!commandsArray.length || typeof commandsArray[0] !== 'string') {
        return [];
    }

    const commands = commandsArray[0].split('\n');
    return commands.map(command => {
        const [type, valueString] = command.split(' ');
        const value = parseFloat(valueString);
        if (!isNaN(value)) {
            return {
                type,
                value: value
            };
        }
    }).filter(command => command !== undefined);
}

async function runPrediction() {

    if (!SERVER_URL) {
        throw new Error("Configuration error: Server URL is not set.");
    }

    const button = document.getElementById('generateButton');
    button.disabled = true;
    button.innerText = "Loading...";

    const rotateBase = Math.floor(Math.random() * 361);
    const prompt = `rotateBase ${rotateBase}`;

    try {
        const currentImageBlob = await captureImageBlob();

        let formData = new FormData();
        if (currentImageBlob) {
            formData.append("file", currentImageBlob, "image.jpg");
        }

        const url = SERVER_URL

        if (!url) {
            throw new Error("Configuration error: Server URL is not set.");
        }

        const response = await fetch(`${url}/process/?prompt=${encodeURIComponent(prompt)}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server response error: status ${response.status}`);
        }

        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error('Error during prediction:', error.message);
        if (error.message.includes('Failed to fetch')) {
            console.log('Network error: Server is unreachable or URL is incorrect.');
        } else if (error.message.includes('Server URL is not set')) {
            console.log('Configuration error: Please set the server URL.');
        } else {
            console.log(`Error: ${error.message}`);
        }
    } finally {
        button.disabled = false;
        button.innerText = "Generate";
    }
}

var saveFile = function (strData, filename) {
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        document.body.appendChild(link);
        link.download = filename;
        link.href = strData;
        link.click();
        document.body.removeChild(link);
    } else {
        location.replace(strData);
    }
}

function saveImage() {
    try {
        forearmRenderer.render(scene, forearmCamera);
        var strMime = "image/jpeg";
        var imgData = forearmRenderer.domElement.toDataURL(strMime, 1.0);
        saveFile(imgData, "forearm_view.jpg");
    } catch (e) {
        console.log(e);
        return;
    }
}

function setupCamera(
    fov = 75,
    aspectRatio = window.innerWidth / window.innerHeight,
    nearClippingPlane = 0.1,
    farClippingPlane = 1000,
    initialPosition = {
        x: 2,
        y: 1,
        z: 0
    },
    lookAtPosition = {
        x: 0,
        y: 0,
        z: 0
    }
) {
    camera = new THREE.PerspectiveCamera(fov, aspectRatio, nearClippingPlane, farClippingPlane);
    camera.position.set(initialPosition.x, initialPosition.y, initialPosition.z);
    camera.lookAt(lookAtPosition.x, lookAtPosition.y, lookAtPosition.z);
}

function setupControls() {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.update();
    document.addEventListener("keydown", onDocumentKeyDown, false);
}

function setupRenderer() {
    renderer = new THREE.WebGLRenderer({
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.gammaFactor = 2.2;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.setClearColor(0x000000);
    document.body.appendChild(renderer.domElement);
}

function setupScene() {
    scene = new THREE.Scene();
    const gridSize = 10;
    const gridDivisions = 10;
    const gridColor = 0x00ff00;
    const centerLineColor = 0xff0000;
    const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, gridColor, centerLineColor);
    //scene.add(gridHelper);
}

document.getElementById('generateButton').addEventListener('click', runPrediction);
document.getElementById('depthMapButton').addEventListener('click', processAndDisplayDepthMap);
document.getElementById('recordButton').addEventListener('click', handleRecordButton);
document.getElementById('saveImageButton').addEventListener('click', saveImage);

window.addEventListener('resize', handleResize);
document.addEventListener("keydown", onDocumentKeyDown);

function onDocumentKeyDown(event) {
    var keyCode = event.which;

    switch (keyCode) {
        case 81:
            robot.rotateBase(1);
            break;
        case 69:
            robot.rotateBase(-1);
            break;
        case 87:
            robot.moveShoulder(-1);
            break;
        case 83:
            robot.moveShoulder(1);
            break;
        case 82:
            robot.moveElbow(-1);
            break;
        case 70:
            robot.moveElbow(1);
            break;
        default:
            break;
    }
}

initRobot('cobot');