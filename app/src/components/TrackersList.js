import React, {
  Component
} from 'react'

import axios from 'axios';

var colors = ['#34495e', '#2ecc71', '#3498db', '#e74c3c', '#d35400', '#34495e', '#ff7a8a', '#ffb142', '#2f3640', '#a64942']


class TrackersList extends Component {
  state = {
    list_ids: [
      [1, false],
      [2, false],
      [3, false],
      [4, false],
      [5, false],
      [6, false]
    ]
  }
  toggleVis(id) {
    this.setState((p) => {
      var list_ids = p.list_ids.slice()
      for (let i = 0; i < list_ids.length; i++) {
        if (list_ids[i][0] == id[0]) {
          list_ids[i][1] = !list_ids[i][1]
        }
      }
      window.excluded_ids = list_ids.filter(e => e[1]).map(e => e[0])
      return {
        list_ids
      }
    })
  }

  render() {
      return ( <
        div className = 'side-bar' >
        <
        h3 > List of Trackers < /h3> {
        this.state.list_ids.map(id => ( < div className = {
              `person ${id[1]?'red':'green'} ${this.state.disabled?'disabled':''}`
            }
            key = {
              id[0]
            }
            onClick = {
              this.toggleVis.bind(this, id)
            } >
            <
            span className = "circle"
            style = {
              {
                backgroundColor: colors[id[0]]
              }
            } > < /span>
            ID({
              id[0]
            }) < /div>))} < /
            div >
          )
        }
      }

      export default TrackersList;