
import * as THREE from "three";

// importing my physics file

import {
    generateDiskPatches,
    orbitalAngularVelocity,
    bulgeCircularVelocity,
    diskCircularVelocity,
    haloCircularVelocity
} from "./physics.js";


// creating scene

const container = document.getElementById("galaxy-canvas");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030407);

const camera = new THREE.PerspectiveCamera(
    45,
    1,
    0.1,
    1000
);

camera.position.set(5.9, 4.4, 11.2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

container.appendChild(renderer.domElement);


// physical parameters

const bulgeMass = 10;
const bulgeScaleRadius = 1.5;

const diskMass = 30;
const diskScaleLength = 2;
const diskRadius = 10;

const gridStep = 0.35;
const softening = gridStep;

const haloScaleDensity = 0.08;
const haloScaleRadius = 5;


// generating disk mass patches

const diskPatches = generateDiskPatches(
    diskMass,
    diskScaleLength,
    diskRadius,
    gridStep
);


// calculating the rotation curve

const rotationCurve = [];
const sampleCount = 100;

for (let i = 0; i < sampleCount; i++) {

    const R = 0.1 + (diskRadius - 0.1) *
        i / (sampleCount - 1);

    const omega = orbitalAngularVelocity(
        R,
        bulgeMass,
        bulgeScaleRadius,
        diskPatches,
        softening,
        haloScaleDensity,
        haloScaleRadius
    );

    rotationCurve.push({
        radius: R,
        omega: omega
    });

    if (i % 20 === 0) {

        const bulgeV = bulgeCircularVelocity(
            R,
            bulgeMass,
            bulgeScaleRadius
        );

        const diskV = diskCircularVelocity(
            R,
            diskPatches,
            softening
        );

        const haloV = haloCircularVelocity(
            R,
            haloScaleDensity,
            haloScaleRadius
        );

        console.log(
            "R:", R.toFixed(2),
            "Bulge:", bulgeV.toFixed(3),
            "Disk:", diskV.toFixed(3),
            "Halo:", haloV.toFixed(3),
            "Total:", Math.sqrt(
                bulgeV ** 2 +
                diskV ** 2 +
                haloV ** 2
            ).toFixed(3)
        );
    }
}


// finding angular velocity between two points in the rotation curve

function getAngularVelocity(R) {

    if (R <= rotationCurve[0].radius) {
        return rotationCurve[0].omega;
    }

    const last = rotationCurve.length - 1;

    if (R >= rotationCurve[last].radius) {
        return rotationCurve[last].omega;
    }

    for (let i = 0; i < last; i++) {

        const a = rotationCurve[i];
        const b = rotationCurve[i + 1];

        if (R >= a.radius && R <= b.radius) {

            const t = (R - a.radius) /
                (b.radius - a.radius);

            return THREE.MathUtils.lerp(
                a.omega,
                b.omega,
                t
            );
        }
    }

    return 0;
}


// generating random values from a Gaussian distribution

function gaussianRandom() {

    let u = 0;
    let v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    return Math.sqrt(-2 * Math.log(u)) *
        Math.cos(2 * Math.PI * v);
}


// selecting star positions according to an exponential disk

function sampleStarRadius(scaleLength, maxRadius) {

    while (true) {

        const R = Math.random() * maxRadius;

        const probability =
            Math.E *
            (R / scaleLength) *
            Math.exp(-R / scaleLength);

        if (Math.random() < probability) {
            return R;
        }
    }
}


// galaxy parameters

const starCount = 50000;
const starColors = new Float32Array(starCount * 3);
const stars = [];

const visualScale = 0.4;

const pitchAngle = THREE.MathUtils.degToRad(18);
const referenceRadius = 1;
const armWidth = 0.22;
const patternSpeed = 0.35;


// creating galaxy group

const galaxy = new THREE.Group();
scene.add(galaxy);


// creating spiral arm reference lines

const spiralPattern = new THREE.Group();
spiralPattern.visible = false;

const armCount = 4;
const armSpacing = (2 * Math.PI) / armCount;
const armPoints = 300;

for (let arm = 0; arm < armCount; arm++) {

    const positions = [];

    for (let i = 0; i < armPoints; i++) {

        const R = 0.2 + (diskRadius - 0.2) *
            i / (armPoints - 1);

        const theta =
            arm * armSpacing +
            Math.log(R / referenceRadius) /
            Math.tan(pitchAngle);

        positions.push(
            R * Math.cos(theta) * visualScale,
            0,
            R * Math.sin(theta) * visualScale
        );
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3)
    );

    const material = new THREE.LineBasicMaterial({
        color: 0xc5b8ff,
        transparent: true,
        opacity: 0.8
    });

    spiralPattern.add(
        new THREE.Line(geometry, material)
    );
}

galaxy.add(spiralPattern);


// tilting the galaxy

galaxy.rotation.z = THREE.MathUtils.degToRad(-20);


// creating the central bulge

const bulgeCount = 12000;
const bulgePositions = new Float32Array(bulgeCount * 3);

for (let i = 0; i < bulgeCount; i++) {

    const radius = Math.pow(Math.random(), 2) * 0.65;

    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.cos(phi) * 0.8;
    const z = radius * Math.sin(phi) * Math.sin(theta);

    bulgePositions[i * 3] = x * visualScale;
    bulgePositions[i * 3 + 1] = y * visualScale;
    bulgePositions[i * 3 + 2] = z * visualScale;
}

const bulgeGeometry = new THREE.BufferGeometry();

bulgeGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(bulgePositions, 3)
);

const bulgeMaterial = new THREE.PointsMaterial({
    color: 0xffe2bd,
    size: 0.024,
    transparent: true,
    opacity: 0.65,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const bulge = new THREE.Points(
    bulgeGeometry,
    bulgeMaterial
);

galaxy.add(bulge);


// central glow

const coreGeometry = new THREE.SphereGeometry(0.8, 32, 32);

const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0xb9d7ff,
    transparent: true,
    opacity: 0.12,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const core = new THREE.Mesh(
    coreGeometry,
    coreMaterial
);

galaxy.add(core);


// light from the central region

const coreLight = new THREE.PointLight(
    0xc5ddff,
    12,
    7,
    2
);

galaxy.add(coreLight);


// creating the disk stars

const starPositions = new Float32Array(starCount * 3);

for (let i = 0; i < starCount; i++) {

    const R = sampleStarRadius(
        diskScaleLength,
        diskRadius
    );

    const theta = Math.random() * Math.PI * 2;

    const x = R * Math.cos(theta);
    const z = R * Math.sin(theta);

    const y = (Math.random() - 0.5) * 0.04;

    starPositions[i * 3] = x * visualScale;
    starPositions[i * 3 + 1] = y * visualScale;
    starPositions[i * 3 + 2] = z * visualScale;

    const roll = Math.random();

    let starColor;

    if (roll < 0.65) {
        starColor = new THREE.Color(0xe6efff);
    }
    else if (roll < 0.85) {
        starColor = new THREE.Color(0xb8d5ff);
    }
    else if (roll < 0.97) {
        starColor = new THREE.Color(0xffffff);
    }
    else {
        starColor = new THREE.Color(0x8ebaff);
    }

    const brightness = 0.35 + Math.random() * 0.65;

    starColors[i * 3] = starColor.r * brightness;
    starColors[i * 3 + 1] = starColor.g * brightness;
    starColors[i * 3 + 2] = starColor.b * brightness;

    stars.push({
        R: R,
        theta: theta,
        initialTheta: theta,
        omega: getAngularVelocity(R)
    });
}

const starGeometry = new THREE.BufferGeometry();

starGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(starPositions, 3)
);

starGeometry.setAttribute(
    "color",
    new THREE.BufferAttribute(starColors, 3)
);

const starMaterial = new THREE.PointsMaterial({
    size: 0.025,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const diskStars = new THREE.Points(
    starGeometry,
    starMaterial
);

galaxy.add(diskStars);


// creating spiral arm particles

const spiralStarCount = 18000;
const spiralPositions = new Float32Array(spiralStarCount * 3);

for (let i = 0; i < spiralStarCount; i++) {

    const R = sampleStarRadius(
        diskScaleLength,
        diskRadius
    );

    const armIndex = Math.floor(
        Math.random() * armCount
    );

    const baseTheta =
        armIndex * armSpacing +
        Math.log(R / referenceRadius) /
        Math.tan(pitchAngle);

    const angularSpread = armWidth / Math.max(R, 0.1);

    const theta =
        baseTheta +
        gaussianRandom() * angularSpread;

    const y = gaussianRandom() * 0.035;

    spiralPositions[i * 3] =
        R * Math.cos(theta) * visualScale;

    spiralPositions[i * 3 + 1] =
        y * visualScale;

    spiralPositions[i * 3 + 2] =
        R * Math.sin(theta) * visualScale;
}

const armGeometry = new THREE.BufferGeometry();

armGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(spiralPositions, 3)
);

const armMaterial = new THREE.PointsMaterial({
    color: 0xc9dcff,
    size: 0.024,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const spiralStarsGroup = new THREE.Group();

const spiralStars = new THREE.Points(
    armGeometry,
    armMaterial
);

spiralStarsGroup.add(spiralStars);


// adding a soft glow to the spiral arms

const glowMaterial = new THREE.PointsMaterial({
    color: 0x8ebaff,
    size: 0.055,
    transparent: true,
    opacity: 0.16,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const spiralGlow = new THREE.Points(
    armGeometry,
    glowMaterial
);

spiralStarsGroup.add(spiralGlow);

galaxy.add(spiralStarsGroup);


// camera and galaxy positioning

camera.lookAt(0, 0, 0);


// cursor interaction

const raycaster = new THREE.Raycaster();

const pointer = new THREE.Vector2();

const pointerPlane = new THREE.Plane(
    new THREE.Vector3(0, 1, 0),
    0
);

const pointerWorld = new THREE.Vector3();
const pointerLocal = new THREE.Vector3();

let currentPointer = null;
let targetPointer = null;

const pointerSmoothness = 1.2;
const distortionSmoothness = 1.5;

const maxDisplacement = 0.25;
const sigma = 1.2;

const canvas = renderer.domElement;

canvas.addEventListener("pointermove", (event) => {

    const rect = canvas.getBoundingClientRect();

    pointer.x =
        ((event.clientX - rect.left) / rect.width) * 2 - 1;

    pointer.y =
        -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);

    const transformedNormal = new THREE.Vector3(0, 1, 0)
        .applyQuaternion(galaxy.quaternion);

    const transformedPlane = new THREE.Plane()
        .setFromNormalAndCoplanarPoint(
            transformedNormal,
            galaxy.position
        );

    if (
        raycaster.ray.intersectPlane(
            transformedPlane,
            pointerWorld
        )
    ) {

        pointerLocal.copy(pointerWorld);

        galaxy.worldToLocal(pointerLocal);

        targetPointer = new THREE.Vector2(
            pointerLocal.x,
            pointerLocal.z
        );

        if (currentPointer === null) {
            currentPointer = targetPointer.clone();
        }
    }
});

canvas.addEventListener("pointerleave", () => {
    targetPointer = null;
});


// animation

const clock = new THREE.Clock();

let patternAngle = 0;

const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
).matches;

const timeScale = reducedMotion ? 0 : 0.25;

let distortionStrength = 0;

const currentPositions = starGeometry.attributes.position;

function animate() {

    requestAnimationFrame(animate);

    const delta = Math.min(clock.getDelta(), 0.05);

    patternAngle += patternSpeed * delta * timeScale;

    spiralStarsGroup.rotation.y = -patternAngle;

    if (targetPointer !== null) {

        if (currentPointer === null) {
            currentPointer = targetPointer.clone();
        }

        currentPointer.lerp(
            targetPointer,
            Math.min(1, pointerSmoothness * delta)
        );

        distortionStrength = THREE.MathUtils.lerp(
            distortionStrength,
            1,
            Math.min(1, distortionSmoothness * delta)
        );

    } else {

        distortionStrength = THREE.MathUtils.lerp(
            distortionStrength,
            0,
            Math.min(1, distortionSmoothness * delta)
        );
    }

    for (let i = 0; i < starCount; i++) {

        const star = stars[i];

        star.theta += star.omega * delta * timeScale;

        const baseX =
            star.R * Math.cos(star.theta);

        const baseZ =
            star.R * Math.sin(star.theta);

        let x = baseX;
        let z = baseZ;

        if (currentPointer !== null) {

            const dx = baseX - currentPointer.x;
            const dz = baseZ - currentPointer.y;

            const distanceSquared = dx * dx + dz * dz;

            const influence = Math.exp(
                -distanceSquared / (2 * sigma * sigma)
            );

            const displacement =
                maxDisplacement *
                influence *
                distortionStrength;

            const distance = Math.sqrt(distanceSquared);

            if (distance > 0.0001) {

                x += (dx / distance) * displacement;
                z += (dz / distance) * displacement;
            }
        }

        currentPositions.array[i * 3] =
            x * visualScale;

        currentPositions.array[i * 3 + 1] =
            starPositions[i * 3 + 1];

        currentPositions.array[i * 3 + 2] =
            z * visualScale;
    }

    currentPositions.needsUpdate = true;

    renderer.render(scene, camera);
}


// adjusting the galaxy when the browser size changes

const resizeObserver = new ResizeObserver(() => {

    const width = container.clientWidth;
    const height = container.clientHeight;

    if (width === 0 || height === 0) return;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
});

resizeObserver.observe(container);


// starting the animation

requestAnimationFrame(() => {
    container.classList.add("ready");
});

animate();
