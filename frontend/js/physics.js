// Gravitational constant in simulation units (normalised, not SI)
const G = 1;

//plummer bulge mass density at a given r

function bulgeDensity(r , mass , scaleRadius){

    /* r is distance from centre of bulge to the particle
    mass - total bulge mass 
    scaleRadius - controls the volume / conc of the bulge
    with all these we compute the density of at that point 
    */
    const a = scaleRadius;
    const density = (3*mass) / (4 * Math.PI * Math.pow(a,3));

    return density * Math.pow(1+ Math.pow(r/a,2), -2.5);
}

//calculating enclosed mass
//total mass enclosed within radius r
function bulgeEnclosedMass(r , mass , scaleRadius){
    // at r =0 enclosed mass is zero , as r increases enclosed mass increases
    // at very long r dist , at a point it becomes equal to the bulge mass M
    const a = scaleRadius;
    const numerator = mass * Math.pow(r,3);
    const denominator = Math.pow(r*r + a*a , 1.5);
    return numerator / denominator
}

//gravitation accelration due to bulge
function bulgeGravitationalAcceleration(r , mass , scaleRadius){

    if (r==0) return 0;


    const enclosedMass = bulgeEnclosedMass(r , mass , scaleRadius);
    return (G * enclosedMass) / (r*r);
}

//circular orbital velocity due to bulge
function bulgeCircularVelocity(r , mass, scaleRadius){
    /*
     for every radius we can calculate the circular speed
     of a star orbiting under the bulge's gravity alone
    */
    const acceleration = bulgeGravitationalAcceleration(r , mass , scaleRadius);
    return Math.sqrt(r * acceleration);

}

//exponential stellar disk , surface mass density

function diskSurfaceDensity( R , diskMass , scaleLength){

    const Rd = scaleLength;
    const centralDensity = diskMass / ( 2 * Math.PI * Rd * Rd);

    return centralDensity * Math.exp(-R/Rd);
}

//mass contained in small patch in a disk
function diskPatchMass( R , diskMass , scaleLength , cellArea){

    const density = diskSurfaceDensity(R , diskMass, scaleLength);
    return density * cellArea;
}

//gravitation accleration from one mass patch

function diskPatchAcceleration(starPosition , patchPosition , patchMass , softening){

    const dx = patchPosition.x - starPosition.x;
    const dy = patchPosition.y - starPosition.y;

    const distancceSquared = dx * dx + dy * dy + softening * softening;

    if (distancceSquared ==0) {
        return { x:0 , y:0};
    }
    const distanceCubed = Math.pow(distancceSquared , 1.5);
    const factor = G * patchMass / distanceCubed;

    return{
        x: factor * dx,
        y: factor * dy
    };
}

//total gravity acceleration from the disk

function diskGravitationalAcceleration(startPosition , diskPatches , softening){

    let totalAx = 0;
    let totalAy = 0; //changes so let is better

    for(const patch of diskPatches){
        const acceleration = diskPatchAcceleration(startPosition,patch.position , patch.mass , softening);
        totalAx += acceleration.x;
        totalAy += acceleration.y;
    }
    return{
        x: totalAx,
        y: totalAy
    };
}

//generating mass patches aross the exponential disk
function generateDiskPatches(diskMass, scaleLength, diskRadius ,step){

    const patches =[];
    const cellArea = step * step;

    // integer indices keep the grid symmetric about the centre (no float drift)
    const n = Math.floor(diskRadius / step);

    for (let i = -n; i <= n; i++) {
        const x = i * step;
        for (let j = -n; j <= n; j++) {
            const y = j * step;
            const R = Math.sqrt(x * x + y * y);
            if(R > diskRadius) continue;

            const mass = diskPatchMass(R , diskMass , scaleLength ,cellArea);
            patches.push({
                position: {x,y},
                mass : mass
            });
        }
    }
    return patches;

}

//circular velocity due to stellar disk

function diskCircularVelocity(R , diskPatches , softening){

    if(R==0) return 0;

    const acceleration = diskGravitationalAcceleration(
        {
            x:R,
            y:0
        },
        diskPatches,
        softening
    );

    const radialAcceleration = -acceleration.x;

    return Math.sqrt(R * Math.max(0 , radialAcceleration));
}



//NFW dark matter halo  density at given radius

function haloDensity( r, scaleDensity , scaleRadius){

    if( r==0) return Infinity;

    const x = r / scaleRadius;

    return scaleDensity / (x * Math.pow(1+x ,2));
}

//total dark matte mass enclosed within radius r

function haloEnclosedMass(r , scaleDensity , scaleRadius){

    if(r ==0) return 0;

    const x = r / scaleRadius;

    const enclosedMass = 4 * Math.PI * scaleDensity * Math.pow(scaleRadius,3) *
    (Math.log(1+x) - x/ (1+x));

    return enclosedMass;
}

// gravitational acceleration due to halo
function haloGravityAcceleration(r , scaleDensity , scaleRadius){
    if(r==0) return 0;


    const enclosedMass = haloEnclosedMass(r, scaleDensity,scaleRadius);
    return ( G * enclosedMass) / ( r*r);
}

// circular velocity due to halo
function haloCircularVelocity(r ,scaleDensity , scaleRadius){
    const acceleration = haloGravityAcceleration(r,scaleDensity,scaleRadius);
    return Math.sqrt( r * acceleration);
}

// Combined circular velocity due to bulge, disk and dark matter halo

function totalCircularVelocity(
    R,
    bulgeMass,
    bulgeScaleRadius,
    diskPatches,
    softening,
    haloScaleDensity,
    haloScaleRadius
){

    const bulgeVelocity = bulgeCircularVelocity(
        R,
        bulgeMass,
        bulgeScaleRadius
    );

    const diskVelocity = diskCircularVelocity(
        R,
        diskPatches,
        softening
    );

    const haloVelocity = haloCircularVelocity(
        R,
        haloScaleDensity,
        haloScaleRadius
    );

    return Math.sqrt(
        bulgeVelocity ** 2 +
        diskVelocity ** 2 +
        haloVelocity ** 2
    );
}

// Orbital angular velocity of a star

function orbitalAngularVelocity(
    R,
    bulgeMass,
    bulgeScaleRadius,
    diskPatches,
    softening,
    haloScaleDensity,
    haloScaleRadius
){

    if(R === 0) return 0;

    const velocity = totalCircularVelocity(
        R,
        bulgeMass,
        bulgeScaleRadius,
        diskPatches,
        softening,
        haloScaleDensity,
        haloScaleRadius
    );

    return velocity / R;
}

//for testing we'll use separate file so export it so that
//we can use
export {
    bulgeDensity,
    bulgeEnclosedMass,
    bulgeGravitationalAcceleration,
    bulgeCircularVelocity,
    diskSurfaceDensity,
    diskPatchMass,
    diskPatchAcceleration,
    diskGravitationalAcceleration,
    generateDiskPatches,
    diskCircularVelocity,
    haloDensity,
    haloEnclosedMass,
    haloGravityAcceleration,
    haloCircularVelocity,
    totalCircularVelocity,
    orbitalAngularVelocity
};