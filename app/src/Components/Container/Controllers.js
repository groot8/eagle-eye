import React, { Component } from 'react'
import axios from 'axios';

class Controllers extends Component {
  state = {
    pause: false
  };


  render() {
    const play_pause = (this.state.pause) ? <button className='play'></button> : <button className='pause'></button>;

    return (
      <div className={this.props.class} onClick={() => {
        this.setState({
          pause: !this.state.pause
        })
        axios.get('http://0.0.0.0:5000/toggle-pause').catch(() => { })
      }
      }>
        <div>
          {play_pause}

        </div>
      </div>
    )
  }
}

export default Controllers;