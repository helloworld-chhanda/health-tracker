#!/bin/bash

# Setup script for AI Health Tracker application

# Print a message with colored text
print_message() {
    echo -e "\e[1;34m>>> $1\e[0m"
}

# Create necessary directories
create_directories() {
    print_message "Creating necessary directories..."
    mkdir -p processed_data
    mkdir -p saved_models
    mkdir -p visualizations
}

# Install Python dependencies
install_dependencies() {
    print_message "Installing Python dependencies..."
    pip install -r requirements.txt
}

# Run the application
run_application() {
    print_message "Running the application..."
    python main.py "$@"
}

# Main function
main() {
    print_message "Setting up AI Health Tracker application..."
    
    # Create directories
    create_directories
    
    # Install dependencies
    install_dependencies
    
    # Ask if the user wants to run the application
    read -p "Do you want to run the application now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Pass all arguments to the main.py script
        run_application "$@"
    else
        print_message "Setup complete! You can run the application later with 'python main.py'"
    fi
}

# Execute main function with all arguments
main "$@" 