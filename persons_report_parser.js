var fs = require('fs')
var data = fs.readFileSync(process.argv[2])
data = data.toString().slice(0, -1).split('\n').map(e => JSON.parse(e));
people_count = data.reduce((a, c) => {
    for (key in c) {
        if (!~a.indexOf(key)) {
            a.push(key)
        }
    }
    return a
}, []).length
var output = ""
for (var j = 1; j < people_count; j++) {
    output += `${j},`
}
output += `${j}\n`
for (var i = 0; i < data.length; i++) {
    for (var j = 0; j < (people_count - 1); j++) {
        if (data[i][j] !== undefined) {       
            output += `[${data[i][j][0]},${data[i][j][1]}],`
        }else{
            output += `-2,`
        }
    }
    if (data[i][j] !== undefined) {       
        output += `[${data[i][j][0]},${data[i][j][1]}]`
    }else{
        output += `-2`
    }
    if (i + 1 != data.length) {
        output += '\n'
    }
}
console.log(output)