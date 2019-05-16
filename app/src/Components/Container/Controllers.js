import React, { Component } from 'react'

 class Controllers extends Component {
    state = {
        pause: false
      };

      
  render() {
    const play_pause = (this.state.pasue)?  <button className='play'></button> :  <button className='pause'></button> ;

    return (
        <div className={this.props.class} onClick={() =>
            this.setState({
              pause: !this.state.pause
            })
            }>
            <div>
               {play_pause}

            </div>
        </div>
    )
  }
}

export default Controllers;