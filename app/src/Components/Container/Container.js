import React, { Component } from 'react'
import './Container.css'
import Stream from './Stream';
import Map from './Map';
import Controllers from './Controllers';


 class Container extends Component {
  state = {
    srcs: [
      'http://0.0.0.0:5000/video_feed?si=0',
      'http://0.0.0.0:5000/video_feed?si=1',
      'http://0.0.0.0:5000/video_feed?si=2',
      'http://0.0.0.0:5000/video_feed?si=3',

    ]
  };

  // src={{uri: this.state.srcs[0]}} 
  render() {
    return (
      <div className='container'>
        <div className='streams'>
          <Stream  src ={this.state.srcs[0]} class = 'stream' />
          <Stream src ={this.state.srcs[1]}  class = 'stream' />
          <Stream src ={this.state.srcs[2]}  class = 'stream' />
          <Stream src ={this.state.srcs[3]}  class = 'stream' />
          {/* <Controllers class='controller' /> */}

        </div>
        {/* <Map src ={require('../../images/map.jpeg')} class = 'map'/> */}
      </div>
    )
  }
}


export default Container;
