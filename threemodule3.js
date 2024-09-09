import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import WebGL from 'three/addons/capabilities/WebGL.js';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { MDDLoader } from 'three/addons/loaders/MDDLoader.js';

let selectedObject = null;
let raycaster = new THREE.Raycaster();
let pointer = new THREE.Vector2();
let scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
let clock = new THREE.Clock();

let renderer = new THREE.WebGLRenderer();
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
let controls = new OrbitControls( camera, renderer.domElement );

let results_path = window.opener.results_path
let umap3d_data = window.opener.umap3d_data
let task_type = window.opener.task_type111
let num_patches1 = window.opener.num_patches1
let final_centroids_data = window.opener.final_centroids_data;
let center_x = window.opener.center_x
let center_y = window.opener.center_y
let center_z = window.opener.center_z
let scale = 10;

var group = new THREE.Object3D();
for (let ind in umap3d_data) {
    let line = umap3d_data[ind];
    let splits = line.split(',');
    let slide_name = splits[0];
    let x = parseFloat(splits[1]);
    let y = parseFloat(splits[2]);
    let z = parseFloat(splits[3]);
    let sprite = createSprite(line, `${results_path}/../../../../patch_images/${slide_name}/x${splits[9]}_y${splits[10]}.JPEG`,
        x-center_x, y-center_y, z-center_z);
    group.add(sprite);
}
scene.add(group);

const materialargs = {
    color: 0xffffff,
    specular: 0x050505,
    shininess: 50,
    emissive: 0x000000
};
const loader = new FontLoader();
loader.load( 'fonts/helvetiker_regular.typeface.json', function ( font ) {
    for (let ind in final_centroids_data) {
        let line = final_centroids_data[ind];
        let splits = line.split(',');
        let c = splits[0];
        let x = parseFloat(splits[1]);
        let y = parseFloat(splits[2]);
        let z = parseFloat(splits[3]);
        const geometry = new THREE.SphereGeometry(1);
        const material = new THREE.MeshBasicMaterial({color: colourNameToHex(CSS_COLOR_NAMES[ind])});
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(scale * (x - center_x), scale * (y - center_y), scale * (z - center_z));
        scene.add(sphere);
    }
});

camera.position.set(0, 0, scale*center_z + 5);
//controls.target.set(center_x, center_y, center_z);
scene.add( new THREE.AxesHelper(10) );
controls.update();
document.addEventListener('click', onPointerClicked);
window.addEventListener('resize', onWindowResize);

function createSprite(line, imgpath, x, y, z) {
    // texture.magFilter = THREE.NearestFilter;
    var texture = new THREE.TextureLoader().load(imgpath);
    var spriteMaterial = new THREE.SpriteMaterial({
        opacity: 0.6,
        transparent: false,
        sizeAttenuation: true,
        map: texture
    });

    // we have one row, with five sprites
    spriteMaterial.blending = THREE.AdditiveBlending;

    var sprite = new THREE.Sprite(spriteMaterial);
    //sprite.scale.set(0.1, 0.1, 0.1);
    sprite.position.set(scale*x, scale*y, scale*z);
    sprite.name = line;

    return sprite;
}

function onPointerClicked( event ) {
    if (event.type !== 'click') {
        return;
    }

    if ( selectedObject ) {
        selectedObject = null;
    }

    pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
    raycaster.setFromCamera( pointer, camera );
    const intersects = raycaster.intersectObject( group, true );
    if ( intersects.length > 0 ) {
        const res = intersects.filter( function ( res ) {
            return res && res.object;
        } )[ 0 ];
        if ( res && res.object ) {
            selectedObject = res.object;
            console.log(selectedObject.name);

            const junk = selectedObject.name.split(',')
            const annoid = junk[junk.length - 1].replace('.jpg', '');
            const slide_name = junk[0].split('/')[1];
            let slide1_name = window.opener.slide1_name
            let slide2_name = window.opener.slide2_name
            let viewer = null;
            if (slide_name === slide1_name) {
                viewer = window.opener.viewer1;
            } else if (slide_name === slide2_name) {
                viewer = window.opener.viewer11;
            } else {
                console.log('slide_name is not matched');
            }
            console.log(`slide1 is ${slide1_name}`);
            console.log(`slide2 is ${slide2_name}`);
            console.log(`you clicked ${slide_name}`);
            if (viewer != null) {
                let coordx = parseInt(junk[6]);
                let coordy = parseInt(junk[7]);
                console.log(coordx, coordy);
                let viewport_coord = viewer.viewport.imageToViewportCoordinates(coordx, coordy);
                viewer.viewport.panTo(viewport_coord);
            }

            if (event.button === 2) {
                console.log("mouse-right clicked")
            }


        } else {
            console.log('res is null')
        }
    } else {
        console.log('no intersects');
    }
}

function onPointerMove( event ) {
    if ( selectedObject ) {
        // selectedObject.material.color.set( '#69f' );
        selectedObject = null;
    }

    pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
    raycaster.setFromCamera( pointer, camera );
    const intersects = raycaster.intersectObject( group, true );
    if ( intersects.length > 0 ) {
        const res = intersects.filter( function ( res ) {
            return res && res.object;
        } )[ 0 ];
        if ( res && res.object ) {
            selectedObject = res.object;
            // selectedObject.material.color.set( '#f00' );
        }
    }
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    controls.update();
    renderer.setSize( window.innerWidth, window.innerHeight );
}

function animate() {
    requestAnimationFrame( animate );

    if (selected_patch_mesh !== null) {
        selected_patch_mesh.rotation.x += 0.01;
        selected_patch_mesh.rotation.y += 0.01;

        const delta = clock.getDelta();
        if ( selected_patch_mixer ) selected_patch_mixer.update( delta );

    }
    controls.update();
    renderer.render( scene, camera );
}


function colourNameToHex(colour)
{
    var colours = {"aliceblue":"#f0f8ff","antiquewhite":"#faebd7","aqua":"#00ffff","aquamarine":"#7fffd4","azure":"#f0ffff",
        "beige":"#f5f5dc","bisque":"#ffe4c4","black":"#000000","blanchedalmond":"#ffebcd","blue":"#0000ff","blueviolet":"#8a2be2","brown":"#a52a2a","burlywood":"#deb887",
        "cadetblue":"#5f9ea0","chartreuse":"#7fff00","chocolate":"#d2691e","coral":"#ff7f50","cornflowerblue":"#6495ed","cornsilk":"#fff8dc","crimson":"#dc143c","cyan":"#00ffff",
        "darkblue":"#00008b","darkcyan":"#008b8b","darkgoldenrod":"#b8860b","darkgray":"#a9a9a9","darkgreen":"#006400","darkkhaki":"#bdb76b","darkmagenta":"#8b008b","darkolivegreen":"#556b2f",
        "darkorange":"#ff8c00","darkorchid":"#9932cc","darkred":"#8b0000","darksalmon":"#e9967a","darkseagreen":"#8fbc8f","darkslateblue":"#483d8b","darkslategray":"#2f4f4f","darkturquoise":"#00ced1",
        "darkviolet":"#9400d3","deeppink":"#ff1493","deepskyblue":"#00bfff","dimgray":"#696969","dodgerblue":"#1e90ff",
        "firebrick":"#b22222","floralwhite":"#fffaf0","forestgreen":"#228b22","fuchsia":"#ff00ff",
        "gainsboro":"#dcdcdc","ghostwhite":"#f8f8ff","gold":"#ffd700","goldenrod":"#daa520","gray":"#808080","green":"#008000","greenyellow":"#adff2f",
        "honeydew":"#f0fff0","hotpink":"#ff69b4",
        "indianred ":"#cd5c5c","indigo":"#4b0082","ivory":"#fffff0","khaki":"#f0e68c",
        "lavender":"#e6e6fa","lavenderblush":"#fff0f5","lawngreen":"#7cfc00","lemonchiffon":"#fffacd","lightblue":"#add8e6","lightcoral":"#f08080","lightcyan":"#e0ffff","lightgoldenrodyellow":"#fafad2",
        "lightgrey":"#d3d3d3","lightgreen":"#90ee90","lightpink":"#ffb6c1","lightsalmon":"#ffa07a","lightseagreen":"#20b2aa","lightskyblue":"#87cefa","lightslategray":"#778899","lightsteelblue":"#b0c4de",
        "lightyellow":"#ffffe0","lime":"#00ff00","limegreen":"#32cd32","linen":"#faf0e6",
        "magenta":"#ff00ff","maroon":"#800000","mediumaquamarine":"#66cdaa","mediumblue":"#0000cd","mediumorchid":"#ba55d3","mediumpurple":"#9370d8","mediumseagreen":"#3cb371","mediumslateblue":"#7b68ee",
        "mediumspringgreen":"#00fa9a","mediumturquoise":"#48d1cc","mediumvioletred":"#c71585","midnightblue":"#191970","mintcream":"#f5fffa","mistyrose":"#ffe4e1","moccasin":"#ffe4b5",
        "navajowhite":"#ffdead","navy":"#000080",
        "oldlace":"#fdf5e6","olive":"#808000","olivedrab":"#6b8e23","orange":"#ffa500","orangered":"#ff4500","orchid":"#da70d6",
        "palegoldenrod":"#eee8aa","palegreen":"#98fb98","paleturquoise":"#afeeee","palevioletred":"#d87093","papayawhip":"#ffefd5","peachpuff":"#ffdab9","peru":"#cd853f","pink":"#ffc0cb","plum":"#dda0dd","powderblue":"#b0e0e6","purple":"#800080",
        "rebeccapurple":"#663399","red":"#ff0000","rosybrown":"#bc8f8f","royalblue":"#4169e1",
        "saddlebrown":"#8b4513","salmon":"#fa8072","sandybrown":"#f4a460","seagreen":"#2e8b57","seashell":"#fff5ee","sienna":"#a0522d","silver":"#c0c0c0","skyblue":"#87ceeb","slateblue":"#6a5acd","slategray":"#708090","snow":"#fffafa","springgreen":"#00ff7f","steelblue":"#4682b4",
        "tan":"#d2b48c","teal":"#008080","thistle":"#d8bfd8","tomato":"#ff6347","turquoise":"#40e0d0",
        "violet":"#ee82ee",
        "wheat":"#f5deb3","white":"#ffffff","whitesmoke":"#f5f5f5",
        "yellow":"#ffff00","yellowgreen":"#9acd32"};

    if (typeof colours[colour.toLowerCase()] != 'undefined')
        return colours[colour.toLowerCase()];

    return false;
}

var selected_patch_mesh = null;
var selected_patch_mixer = null;
// Called sometime after postMessage is called
window.addEventListener("message", (event) => {
    console.log("yes, accept an message from ", event.origin);
    console.log("event.data is ", event.data);

    let junk = event.data.split(',');
    let x = scale* (parseFloat(junk[1]) - center_x);
    let y = scale* (parseFloat(junk[2]) - center_y);
    let z = scale* (parseFloat(junk[3]) - center_z);
    console.log(x, y, z);

    if (selected_patch_mesh !== null) {
        scene.remove(selected_patch_mesh);
        selected_patch_mixer = null;
    }

    const loader = new MDDLoader();
    loader.load( 'cube.mdd', function ( result ) {

        const morphTargets = result.morphTargets;
        const clip = result.clip;
        // clip.optimize(); // optional

        const geometry1 = new THREE.BoxGeometry();
        geometry1.morphAttributes.position = morphTargets; // apply morph targets
        var texture1 = new THREE.TextureLoader().load(junk[0]);
        texture1.anisotropy = renderer.capabilities.getMaxAnisotropy();
        var material1 = new THREE.MeshBasicMaterial( { map: texture1 } );
        selected_patch_mesh = new THREE.Mesh( geometry1, material1 );
        selected_patch_mesh.position.set(x, y, z);
        scene.add( selected_patch_mesh );
        selected_patch_mixer = new THREE.AnimationMixer( selected_patch_mesh );
        selected_patch_mixer.clipAction( clip ).play();

    } );
});

if (WebGL.isWebGLAvailable()) {
    animate();
} else {
    const warning = WebGL.getWebGLErrorMessage();
    document.body.appendChild(warning);
}