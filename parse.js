'use strict';

const fs = require('fs');

let rawdata = fs.readFileSync('report.example.json');  
let pos = JSON.parse(rawdata);  
// console.log(pos.length); 
// console.log(pos)

for (let i = 0; i < pos.length; i++) {
    if (Object.entries(pos[i]).length === 0) {
        
        delete pos[i];
    }
}
var filteredPos = pos.filter(function (el) {
    return el != null;
  });

console.log(filteredPos);
// Object.entries(obj).length === 0 && obj.constructor === Object


// var positions = {};
// let keys = []
//  for (let i = 0; i < filteredPos.length; i++) {
//   keys.append(Object.keys(filteredPos[i]));
 
  
//   for (let j= 0; j <= keys.length; j++) {
//       console.log(keys[j])
   
      
//   }
//  };

//  console.log(positions);