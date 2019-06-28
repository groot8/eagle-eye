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
    output += `${j - 1};`
}
output += `${j - 1}\n`
for (var i = 75; i < data.length - 1; i++) {
    if ((i - 75) % 25 != 0) continue
    for (var j = 0; j < people_count - 1; j++) {
        if (data[i][j + 1] === undefined) {
            output += "-2;"
        } else {
            output += `[${data[i][j + 1][0]},${data[i][j + 1][1]}];`
        }
    }
    if (data[i][j + 1] === undefined) {
        output += "-2"
    } else {
        output += `[${data[i][j + 1][0]},${data[i][j + 1][1]}]`
    }
    output += '\n'
}
for (var j = 0; j < people_count - 1; j++) {
    if (data[i][j + 1] === undefined) {
        output += "-2;"
    } else {
        output += `[${data[i][j + 1][0]},${data[i][j + 1][1]}];`
    }
}
if (data[i][j + 1] === undefined) {
    output += "-2"
} else {
    output += `[${data[i][j + 1][0]},${data[i][j + 1][1]}]`
}
console.log(output)